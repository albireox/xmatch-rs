from xmatch_rs.xmatch import XMatch


xm = XMatch(
    "/mnt/mako-big/gallegoj/xmatch_test_data/xmatch-2.0.0",
    version="2.0.0",
    pixelation_order=6,
    epoch=2016.0,
)

xm.xmatch_catalogue(
    "legacy_survey_dr10",
    "/mnt/mako-big/gallegoj/xmatch_test_data/legacy_survey_dr10/",
    run_id=15,
    verbose=True,
    overwrite=True,
    # trace=True,
    join_paths=[("gaia_dr3_source_id", "gaia_dr3_source.source_id")],
)
