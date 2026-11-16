/*
 *  @Author: José Sánchez-Gallego (gallegoj@uw.edu)
 *  @Date: 2026-11-09
 *  @Filename: dataset.rs
 *  @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
 */

use std::{fs::File, path::PathBuf, sync::RwLock};

use anyhow::{Result, anyhow, bail, ensure};
use itertools::Itertools;
use polars::prelude::*;

use crate::{
    hats::tools::{get_dataset_directory, get_hats_path_root, get_pixel_path},
    healpix::healpix_convert_order,
    tools::to_pl_path,
};

// Lock to prevent multiple threads from accessing the same partition file at the same time.
// Not totally sure why this happens but it causes SIGSEGV errors.
static ACCESS_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[derive(Clone, Debug)]
pub struct PartitionInfo {
    _df: DataFrame,
    _file_path: Option<PathBuf>,
}

impl PartitionInfo {
    /// Creates a `PartitionInfo` from a `DataFrame`.
    pub fn from(_df: DataFrame) -> Self {
        Self {
            _df,
            _file_path: None,
        }
    }

    /// Creates a `PartitionInfo` from a CSV file.
    pub fn from_file(path: &PathBuf) -> Result<Self> {
        let fields = vec![
            Field::new("Norder".into(), DataType::UInt8),
            Field::new("Npix".into(), DataType::UInt64),
        ];

        let schema = Schema::from_iter(fields);

        let lf = LazyCsvReader::new(to_pl_path(path)?)
            .with_has_header(true)
            .with_separator(b',')
            .with_schema(Some(Arc::new(schema)))
            .finish()?
            .sort(vec!["Norder", "Npix"], Default::default());

        let _lock = ACCESS_MUTEX.lock().unwrap();

        Ok(Self {
            _df: lf.collect()?,
            _file_path: Some(path.clone()),
        })
    }

    /// Creates a `PartitionInfo` from a dataset path.
    ///
    /// Reads the `partition_info.csv` file if it exists, or generates the partition info from the
    /// dataset directory structure if the file is not found.
    pub fn from_dataset(path: &PathBuf, read_csv: bool) -> Result<Self> {
        let dataset_path = get_dataset_directory(path)?;

        if read_csv {
            let partition_info_path = dataset_path.join("../partition_info.csv");
            if partition_info_path.exists() {
                return Self::from_file(&partition_info_path);
            }
        }

        // Loop over all the pixel directories in the dataset and extract the Norder and Npix
        // from the directory structure. We don't include the Dir column since it doesn't seem
        // to be part of the HATS partition_info schema.
        let mut norder_values: Vec<u32> = Vec::new();
        let mut npix_values: Vec<u32> = Vec::new();

        for entry in glob::glob(&format!("{}/Norder=*/Dir=*/Npix=*", dataset_path.display()))? {
            let entry = entry?;

            let parts: Vec<_> = entry
                .components()
                .map(|c| c.as_os_str().to_string_lossy())
                .collect();

            let norder_part = parts.iter().find(|p| p.starts_with("Norder=")).unwrap();
            let npix_part = parts.iter().find(|p| p.starts_with("Npix=")).unwrap();

            let norder: u32 = norder_part[7..].parse()?;
            let npix: u32 = npix_part[5..].parse()?;

            norder_values.push(norder);
            npix_values.push(npix);
        }

        let lf = df!(
            "Norder" => norder_values,
            "Npix" => npix_values,
        )?
        .lazy()
        .select([
            col("Norder").cast(DataType::UInt8),
            col("Npix").cast(DataType::UInt64),
        ])
        .sort(vec!["Norder", "Npix"], Default::default());

        let _lock = ACCESS_MUTEX.lock().unwrap();

        Ok(Self {
            _df: lf.collect()?,
            _file_path: None,
        })
    }

    /// Writes the partition info to a CSV file.
    pub fn to_csv(&self, path: &PathBuf) -> Result<()> {
        let mut df = self
            .get()
            .lazy()
            .sort(vec!["Norder", "Npix"], Default::default())
            .collect()?;

        let pl_path = to_pl_path(path)?;
        CsvWriter::new(File::create(pl_path)?)
            .with_separator(b',')
            .include_header(true)
            .finish(&mut df)?;

        Ok(())
    }

    /// Gets the data as a `DataFrame`.
    pub fn get(&self) -> DataFrame {
        self._df.clone()
    }

    /// Returns a `LazyFrame` with the HATS data for a pixel, if it exists in the partition.
    pub fn get_pixel(&self, dataset_path: PathBuf, order: u8, npix: u64) -> Result<LazyFrame> {
        if !self.contains(order, npix)? {
            return Err(anyhow!("Pixel not found in the partition"));
        }

        let pixel_path = get_pixel_path(&dataset_path, order, npix);
        let pl_path = to_pl_path(&pixel_path)?;
        let lf = LazyFrame::scan_parquet(pl_path, Default::default())?;

        Ok(lf)
    }

    /// Returns the list of orders in the partition info.
    pub fn orders(&self) -> Result<Vec<u8>> {
        let orders = self.get().lazy().select([col("Norder")]).collect()?;
        let orders_col = orders.column("Norder")?.u8()?;
        Ok(orders_col.into_no_null_iter().collect())
    }

    /// Lists the  pixels in the partition info for a given order/
    pub fn list_order_pixels(&self, order: u8) -> Result<Vec<u64>> {
        Ok(self
            .get()
            .lazy()
            .filter(col("Norder").eq(lit(order)))
            .select([col("Npix")])
            .sort(["Npix"], Default::default())
            .collect()?[0]
            .u64()?
            .into_no_null_iter()
            .collect::<Vec<u64>>())
    }

    /// Number of pixels in the partition info.
    pub fn len(&self) -> usize {
        self._df.height()
    }

    /// Returns the minimum order found in the partition info.
    pub fn min_order(&self) -> Result<u8> {
        let min_order = self.get().lazy().select([col("Norder").min()]).collect()?;

        min_order
            .column("Norder")?
            .u8()?
            .get(0)
            .ok_or_else(|| anyhow!("No value in Norder column"))
    }

    /// Returns the maximum order found in the partition info.
    pub fn max_order(&self) -> Result<u8> {
        let max_order = self.get().lazy().select([col("Norder").max()]).collect()?;

        max_order
            .column("Norder")?
            .u8()?
            .get(0)
            .ok_or_else(|| anyhow!("No value in Norder column"))
    }

    /// Checks if the partition info contains a given order and pixel.
    pub fn contains(&self, order: u8, npix: u64) -> Result<bool> {
        let filtered = self
            .get()
            .lazy()
            .filter(col("Norder").eq(lit(order)).and(col("Npix").eq(lit(npix))));

        let df = filtered.collect()?;
        Ok(df.height() > 0)
    }

    /// Did we read the partition info from a file?
    pub fn was_read_from_file(&self) -> bool {
        self._file_path.is_some()
    }
}

pub struct PartitionInfoIter {
    data: std::vec::IntoIter<(u8, u64)>,
}

impl Iterator for PartitionInfoIter {
    type Item = (u8, u64);

    fn next(&mut self) -> Option<Self::Item> {
        self.data.next()
    }
}

impl IntoIterator for PartitionInfo {
    type Item = (u8, u64);
    type IntoIter = PartitionInfoIter;

    fn into_iter(self) -> Self::IntoIter {
        let pixels = self.get().clone();
        let norder = pixels
            .column("Norder")
            .unwrap()
            .u8()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<u8>>();
        let npix = pixels
            .column("Npix")
            .unwrap()
            .u64()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<u64>>();

        let data: Vec<(u8, u64)> = norder.into_iter().zip(npix).collect();

        PartitionInfoIter {
            data: data.into_iter(),
        }
    }
}

/// A trait representing a dataset.
///
/// This trait provides methods for accessing and interacting with a dataset. Although there
/// is an expectation that the underlying dataset is HATS-formatted, the only requirement is that
/// the pixels must be partitioned following the HATS directory structure
/// `Norder=<order>/Dir=<dir>/Npix=<npix>`.
///
pub trait AsDataset {
    /// Returns the path to the dataset.
    fn path(&self) -> PathBuf;

    /// Returns the dataset as a lazy frame.
    fn get_frame(&self) -> Result<LazyFrame> {
        let mut frame = LazyFrame::scan_parquet(
            self.path().clone().to_str().unwrap().into(),
            ScanArgsParquet::default(),
        )
        .unwrap();

        // Check that the Npix column is present in the dataset.
        ensure!(
            frame.collect_schema()?.contains("Npix"),
            "Npix column not found."
        );

        Ok(frame)
    }

    /// Returns the dataset partition.
    fn partition(&self) -> PartitionInfo {
        PartitionInfo::from_dataset(&get_dataset_directory(&self.path()).unwrap(), true).unwrap()
    }

    /// Gets the path for a pixel directory in the dataset.
    fn get_pixel_path(&self, norder: u8, npix: u64) -> Result<PathBuf> {
        let dataset_dir = get_dataset_directory(&self.path())?;
        Ok(get_pixel_path(&dataset_dir, norder, npix))
    }

    /// Returns a list of neighbours for a given order-pixel.
    ///
    /// Given a dataset and a pixel in a certain order, finds its neighbours in any order within the
    /// dataset. A neighbour, in this context, is defined as any pixel in the dataset that
    /// intersects any of the neighbouring pixels of ``(norder, npix)`` (excluding the central pixel
    /// unless ``include_centre`` is true).
    ///
    /// # Arguments
    /// * `norder`: The order of the pixel for which to find neighbours.
    /// * `npix`: The pixel number for which to find neighbours, within the given order.
    /// * `include_centre`: Whether to include the central pixel.
    ///
    fn get_neighbours(
        &self,
        norder: u8,
        npix: u64,
        include_centre: bool,
    ) -> Result<Vec<(u8, u64)>> {
        let partition_info = self.partition();

        let dataset_min_order = partition_info.min_order()?;
        let dataset_max_order = partition_info.max_order()?;

        // Get the list of neighbours in the requested order.
        let layer_order = cdshealpix::nested::get(norder);
        let cds_neighbours = layer_order.neighbours(npix, include_centre);
        let native_neighbours = cds_neighbours
            .entries()
            .into_iter()
            .map(|(_, npix)| npix)
            .collect::<Vec<_>>();

        // Vector of neighbours in the dataset.
        let mut neighbours: Vec<(u8, u64)> = Vec::new();

        // Iterate over the dataset orders. For each one, retrieves the list of pixels in the
        // dataset and compares it with the list of neighbours, converted to the dataset order.
        for dataset_order in dataset_min_order..=dataset_max_order {
            // List of pixels in the current order.
            let order_pixels = partition_info.list_order_pixels(dataset_order)?;

            // Convert the neighbours to the current dataset order.
            let test_pixels = native_neighbours
                .iter()
                .flat_map(|np| healpix_convert_order(*np, norder, dataset_order))
                .unique()
                .collect::<Vec<_>>();

            // Check if any of the converted pixels are in the partition info for the dataset.
            order_pixels.iter().for_each(|p| {
                if test_pixels.contains(&p) {
                    neighbours.push((dataset_order, *p));
                }
            });
        }

        // Sort by order, then pixel.
        neighbours.sort();
        neighbours.dedup();

        Ok(neighbours)
    }

    /// Returns a `LazyFrame` with the entries in the dataset within a pixel plus buffer area.
    ///
    /// The buffer area extends around the specified HEALPix pixel by the given buffer distance
    /// in arcsec. This assumes that the dataset include the HATS `_healpix_29` column, which
    /// is used to calculate the RA/Dec coordinates of each entry.
    ///
    /// # Arguments
    ///
    /// * `order` - The HEALPix order of the pixel.
    /// * `pixel` - The HEALPix pixel index.
    /// * `buffer` - The buffer distance in arcsec.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `LazyFrame` for the dataset entries within the selected region.
    fn get_pixel_with_buffer(&self, order: u8, pixel: u64, buffer: f32) -> Result<LazyFrame> {
        let schema = self.scan_pixel(order, pixel, true)?.collect_schema()?;
        if !schema.contains("_healpix_29") {
            bail!("Missing column: _healpix_29");
        }

        // Get the BMOC covering the margin around the cell.
        let bmoc = crate::hats::margin::get_margin_bmoc(order, pixel, buffer, None);

        // TODO: this will fail to get all the necessary pixels if the margin is large enough to
        // include pixels that are not direct neighbours. A better way would be to keep including
        // concentric rings of neighbours. When none of the neighbours of a neighbour have
        // intersection with the BMOC, we stop iteraing for that neighbour. For the purposes of
        // creating a margin to account for proper motions, and if the pixelation order is small
        // enough, what we do now is probably sufficient.
        let neighbours = self.get_neighbours(order, pixel, true)?;

        // Iterate over the neighbours and collect the targets that fall within the margin.
        let mut frames: Vec<LazyFrame> = vec![];

        for (neigh_order, neigh_npix) in neighbours {
            // Read neighbour pixel data.
            let n_data = self.scan_pixel(neigh_order, neigh_npix, false)?;

            // We need to clone the BMOC because it is moved inside the closure.
            // BMOC doesn't have clone so we create a new empty BMOC and then use the `or` method
            // to combine it with the original one.
            let bmoc_clone = cdshealpix::nested::bmoc::BMOC::new_empty(12).or(&bmoc);

            // Add a column to the neighbour data indicating whether each target falls within
            // the margin or not. For this we use the _healpix_29 column which we can use to
            // calculate the RA/Dec and then test if it is included in the BMOC.
            let margin_n_data =
                n_data.with_column(
                    col("_healpix_29")
                        .map(
                            move |healpix_29_col| {
                                let radec: Vec<bool> = healpix_29_col
                                    .u64()
                                    .unwrap()
                                    .iter()
                                    .map(|healpix_29| {
                                        let healpix_29 = healpix_29.unwrap();
                                        let (ra, dec) = cdshealpix::nested::center(29, healpix_29);
                                        let status = bmoc_clone.test_coo(ra, dec);
                                        match status {
                                            cdshealpix::nested::bmoc::Status::IN => true,
                                            cdshealpix::nested::bmoc::Status::UNKNOWN => true, // Better safe than sorry.
                                            _ => false,
                                        }
                                    })
                                    .collect();

                                Ok(Series::new(PlSmallStr::from("in_margin_bmoc"), radec)
                                    .into_column())
                            },
                            |_, field| Ok(Field::new(field.name().clone(), DataType::Boolean)),
                        )
                        .alias("in_margin_bmoc"),
                );

            frames.push(margin_n_data);
        }

        // Concatenate the frames for the neighbours and filter to keep only the targets that fall within the margin.
        let margin_lf = concat(frames, Default::default())?
            .filter(col("in_margin_bmoc"))
            .drop(Selector::ByName {
                names: Arc::new(["in_margin_bmoc".into()]),
                strict: true,
            });

        Ok(margin_lf)
    }

    /// Returns a lazy frame for a HEALPix pixel in the dataset.
    ///
    /// # Arguments
    ///
    /// * `order` - The HEALPix order of the pixel.
    /// * `pixel` - The HEALPix pixel index.
    /// * `nearest` - If the order-pixel pair does not exist in the dataset, returns the data from
    ///     all the pixels in the dataset that intersect with the requested pixel. This may result
    ///     in a frame containing more data than requested.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `LazyFrame` for the specified HEALPix pixel and, optionally,
    /// its neighbours.
    ///
    fn scan_pixel(&self, order: u8, pixel: u64, nearest: bool) -> Result<LazyFrame> {
        let partition = self.partition();

        // If the order-pixel is not naturally in the dataset, find the nearest pixels.
        if !partition.contains(order, pixel)? {
            if !nearest {
                bail!("Pixel ({order}, {pixel}) is not in the dataset partition");
            }

            let dataset_min_order = partition.min_order()?;
            let dataset_max_order = partition.max_order()?;

            let mut pixels: Vec<(u8, u64)> = vec![];
            for dataset_order in dataset_min_order..=dataset_max_order {
                let test_pixels = healpix_convert_order(pixel, order, dataset_order);
                for test_pixel in test_pixels {
                    if partition.contains(dataset_order, test_pixel)? {
                        pixels.push((dataset_order, test_pixel));
                    }
                }
            }

            if pixels.is_empty() {
                bail!("No pixels intersect with ({order}, {pixel})");
            }

            return self.scan_pixels(pixels);
        }

        self.scan_pixels(vec![(order, pixel)])
    }

    /// Scans multiple order-pixel pairs in the dataset.
    ///
    /// # Arguments
    ///
    /// * `pixels` - The order-pixel pairs to scan. All pixels must be in the dataset partition.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `LazyFrame` for the specified pixels.
    ///
    fn scan_pixels(&self, pixels: Vec<(u8, u64)>) -> Result<LazyFrame> {
        // Instead of reading the entire dataset and filtering, we scan only the requested
        // pixel files and then concatenate them.

        let mut frames: Vec<LazyFrame> = Vec::new();

        let partition = self.partition();

        for (order, pixel) in pixels {
            if !partition.contains(order, pixel)? {
                bail!("Pixel ({order}, {pixel}) is not in the dataset partition");
            }

            let path = self.get_pixel_path(order, pixel)?;
            let frame =
                LazyFrame::scan_parquet(path.to_str().unwrap().into(), ScanArgsParquet::default())?;
            frames.push(frame);
        }

        Ok(concat(frames, UnionArgs::default())?)
    }
}

#[derive(Debug, Clone)]
pub struct Dataset {
    dataset_path: PathBuf,

    // Thread-safe.
    _partition_cache: Arc<RwLock<Option<PartitionInfo>>>,
}

impl Dataset {
    pub fn new(dataset_path: PathBuf, is_hats: bool) -> Self {
        if !dataset_path.exists() {
            panic!("Dataset path does not exist: {}", dataset_path.display());
        }

        let dataset_path = if is_hats {
            get_hats_path_root(dataset_path.clone())
                .unwrap()
                .join("object/dataset")
        } else {
            dataset_path.clone()
        };

        Self {
            dataset_path,
            _partition_cache: Arc::new(RwLock::new(None)),
        }
    }
}

impl AsDataset for Dataset {
    fn path(&self) -> PathBuf {
        self.dataset_path.clone()
    }

    fn partition(&self) -> PartitionInfo {
        let mut cache = self._partition_cache.write().unwrap();

        if let Some(partition) = cache.as_ref() {
            return partition.clone();
        }

        let partition = PartitionInfo::from_dataset(&self.dataset_path, true).unwrap();
        *cache = Some(partition.clone());

        partition
    }
}
