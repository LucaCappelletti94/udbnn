from udbnn import run, clear, collect

def test_run():
    run("test_dataset")
    collect("test_dataset")
    clear("test_dataset")