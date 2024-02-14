import numpy as np 
from struct import unpack

class PNG():
    #signature
    _PNG_SIGNATURE_LENGTH = 8
    _PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
    
    #IHDR chunk, header
    _PNG_IHDR_LENGTH = 13
    _PNG_IHDR_ITEMS = ('width', 'height', 'bitdepth', 'colortype', 'compress', 'filter', 'interlace')
    _IHDR_type = np.dtype({'names'  : _PNG_IHDR_ITEMS,
                           'formats': ['>u4', '>u4', 'u1', 'u1', 'u1',  'u1', 'u1']}, align=True)
    _CHUNK_LIST = [b"IHDR", b"PLTE", b"tRNS", b"IDAT", b"IEND"] #to be extent
    color_type_name = {0:"Greyscale", 1:"Truecolor", 2:"Indexed-color", 4:"Greyscale with alpha", 6:"Truecolor with alpha"}
    
    #error list
    _private_error = ("no error", "wrong signature", "wrong header", "corrupted data")
    _ERROR = dict(zip(_private_error, range(len(_private_error))))
    
    #tools function
    freadBE32 = lambda finput: unpack(">I", finput.read(4))[0]
    readBE32  = lambda x: unpack(">I", x)[0]
    
    def __init__(self, finput) -> None:
        self.im = finput 
        self.IHDR = np.zeros(1, dtype = PNG._IHDR_type)    
        
    
    @classmethod
    def _getError(cls, errorCode):
        return cls._private_error[errorCode]
    
    """
    #filters, it take a scanline of src, with context (previous scanline) and output in given dst.
    #all should have unit8 dtype.
    """
    @staticmethod
    def None_I(dst:np.ndarray, context:np.ndarray, src:np.ndarray) -> int:
        dst[:] = src
        return 0
    
    @staticmethod
    def Sub_I(dst:np.ndarray, context:np.ndarray, src:np.ndarray) -> int:
        dst[:] = np.cumsum(src, dtype = np.uint8) #must specify its uint8!
        return 0
    
    @staticmethod
    def Up_I(dst:np.ndarray, context:np.ndarray, src:np.ndarray) -> int:
        dst[:] = src + context
        return 0
    
    @staticmethod
    def Average_I(dst:np.ndarray, context:np.ndarray, src:np.ndarray) -> int:
        dst[0] = (context[0]>>1) + src[0]
        for i in range(1, len(src)):
            x = src[i].astype(np.int16)     #current
            b = context[i].astype(np.int16) #upper
            a = dst[i-1].astype(np.int16)   #prev
            dst[i] = (x + ((a + b)>>1))
        return 0
    
    @staticmethod
    def Paeth_I(dst:np.ndarray, context:np.ndarray, src:np.ndarray) -> int:
        dst[0] = src[0].astype(np.int16) + context[0].astype(np.int16)
        for i in range(1, len(src)):
            x = src[i].astype(np.int16)   
            a = dst[i-1].astype(np.int16)   
            b = context[i].astype(np.int16)
            c = context[i-1].astype(np.int16)
            pr = a + b - c
            pa = np.abs(pr - a)
            pb = np.abs(pr - b)
            pc = np.abs(pr - c)
            if pa<= pb and pa <= pc:
                dst[i] = (x+a)
            elif pb <= pc:
                dst[i] = x+b
            else:
                dst[i] = x+c
        return 0



    
    




    




        
    

