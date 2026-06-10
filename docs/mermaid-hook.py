# see https://github.com/danielfrg/mkdocs-jupyter/issues/241
def on_page_content(html, page, config, files):
    return html.replace('class="jp-Mermaid"', 'class="mermaid"')
