import TEST

def test_add():
    assert TEST.add(7, 3) ==10
    assert TEST.add(2) == 4

def test_product():
    assert TEST.product(5,5) == 25
    assert TEST.product(5) == 10
    