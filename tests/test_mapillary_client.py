def test_fetch_metadata_bbox(fake_mapillary_client):
    df = fake_mapillary_client.fetch_metadata_bbox((4.89, 52.37, 4.91, 52.38))

    assert not df.empty
    assert "geometry" in df.columns
    assert df.iloc[0]["id"] == "1"
