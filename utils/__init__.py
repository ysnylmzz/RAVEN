# A regular package, unlike the repo's other top-level directories: a namespace
# package loses to any plain ``utils`` module anywhere on sys.path (e.g. from
# site-packages or CANN's PYTHONPATH on Ascend hosts), whereas a regular package
# in the repo root -- sys.path[0] under ``python -m`` -- is found first.
