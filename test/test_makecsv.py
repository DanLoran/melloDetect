from ..dataset.makecsv import MakeCSV

def test_MakeCsvEmptyList():
    imageList = ['img1', 'img2']
    labelList = []
    assert MakeCSV(labelList, imageList) == 0

def test_MakeCSVWrongFormat():
    imageList = ['img1', 'img2']
    labelList = ['l1', 'l2']
    assert MakeCSV(labelList, imageList) == 0
