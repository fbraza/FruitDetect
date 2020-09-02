import numpy as np


class LabelTreatment:
    """
        Class to handle the treatment of Label Files for input to Albumentation library
        For the moment works with the "yolo" label file format
    """

    def __init__(self:object) -> None:
        """
            Default Initialisation 
        """
        self.fromFile = ''
        self.toFile = ''
        self.labelType = 'yolo'

        """
    Function defined to read an image and output it.
    Args:
        path: string of the image path
        opencv_color_flag:  choose -1 (load image as such) , 0 (loads image in grayscale) or 1 (loads image in BGR)
    Returns:
        np.ndarray
    """

        
    def __init__(self:object, fromFile:str, toFile:str, labelType:str) -> None:
        """
        Initialisation with the source file and the destination file for labels
        and the labelType
        
        Args:
            fromFile : the full path + filename is expected.
            toFile : the full path + filename is expected
            labelType : for the moment only the 'yolo' label file format is treated
        """
        self.fromFile = fromFile
        self.toFile = toFile
        
        # if the labelType is not 'yolo' we quit the programm
        if labelType != 'yolo':
            raise Exception(f'only yolo format label is possible for the moment : {labelType} format not known')
        self.labelType = labelType

        
    def getBoxObjectFromFile(self:object) -> tuple:
        """
        Read the label format file and return a Box list of list that can be used with Albumentation library
        
        Returns:
            tuple : with :
                * a list of list : one line for each box with 4 coordinates
                * a list : with one label for each box
        """
        try:
            box = np.loadtxt(self.fromFile)
        except IOError:
            raise Exception(f'Error when opening {self.fromFile}')
        return (box[:,1:].tolist(),[str(int(i)) for i in (box[:,0].tolist())])
        
    def putBoxObjectToFile(self:object,box:list,labelClass:list) -> None:
        """
        Write a yolo format file from box list and label class list
        
        Args:
            box : a list of list : one line for each box with 4 coordinates
            labelClass : a list with one label for each box
        """
        try:
            f= open(self.toFile,"w")
        except IOError:
            raise Exception(f'Error when opening {self.toFile}')
        
        for i,label in enumerate(labelClass):
            f.write(f"{int(label)} {' '.join([str(coord) for coord in box[i]])}\n")
        f.close()
