from cyclopts import App


segment_images_cli = App(help="Segment images")

segment_images_cli.command("streetscapes.models.bfms.cli:cli", name="bfms")
segment_images_cli.command("streetscapes.models.maskformer.cli:cli", name="maskformer")
segment_images_cli.command("streetscapes.models.dinosam.cli:cli", name="dinosam")
