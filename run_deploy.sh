rm -rf ./_build
jupyter-book build ../nlp-book
ghp-import -n -p -f _build/html
