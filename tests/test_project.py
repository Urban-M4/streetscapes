import pandas as pd
import pytest

from streetscapes.project import Project

def test_ingest_and_filter_bbox(tmp_path, fake_mapillary_data):
    db_path = tmp_path / "test.duckdb"
    project = Project(db_path=str(db_path))

    project.ingest_mapillary(fake_mapillary_data)

    # Filter within bbox
    bbox = (4.88, 52.36, 4.90, 52.38)
    filtered = project.filter_bbox("mapillary", bbox)

    count = filtered.count().execute()
    assert count == 1

def test_export_csv_parquet(tmp_path, fake_mapillary_data):
    """Test that the Project can export CSV and Parquet from a table."""
    db_path = tmp_path / "project.duckdb"
    project = Project(db_path=str(db_path))

    project.ingest_mapillary(fake_mapillary_data)

    # CSV export
    csv_file = tmp_path / "table.csv"
    project.export_csv("mapillary", str(csv_file))
    assert csv_file.exists()
    assert csv_file.stat().st_size > 0

    # Parquet export
    parquet_file = tmp_path / "table.parquet"
    project.export_parquet("mapillary", str(parquet_file))
    assert parquet_file.exists()
    # Read back Parquet
    df_out = pd.read_parquet(parquet_file)
    assert len(df_out) == 2
    assert "geometry" in df_out.columns


def test_project_export_requires_geometry(tmp_path, fake_mapillary_data):
    """Test that exporting geospatial formats raises if geometry is missing."""
    db_path = tmp_path / "project.duckdb"
    project = Project(db_path=str(db_path))

    project.ingest_mapillary(fake_mapillary_data)
    project.db.raw_sql("ALTER TABLE mapillary DROP geometry")

    # GeoPackage
    with pytest.raises(ValueError, match="requires a 'geometry'"):
        project.export_gpkg("mapillary", str(tmp_path / "out.gpkg"))

    # GeoJSON
    with pytest.raises(ValueError, match="requires a 'geometry'"):
        project.export_geojson("mapillary", str(tmp_path / "out.geojson"))