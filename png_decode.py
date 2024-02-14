from lib_png import PNG
import zlib
import numpy as np

class PNG_decoder(PNG):
    filter_type =(PNG.None_I, PNG.Sub_I, PNG.Up_I, PNG.Average_I, PNG.Paeth_I)
    filter_dict = dict(enumerate(filter_type))
    _private_error = ("no error", "zlib decompress fail")
    _DERROR = dict(zip(_private_error, range(len(_private_error))))
    def __init__(self, finput) -> None:
        super().__init__(finput)
        self.topology = []
        self.offset = []
        try:
            check = self._getChunkLayout()
            if check !=0: 
                raise RuntimeError
        except RuntimeError:
            print("can not get chunk layout:", PNG._getError(check))
        self.bands = self._getChannelNb()
        self.pixelwidth = self._getPixelbyteNb() #bands*2 if 16 = bitdepth

    def _getChunkLayout(self) -> int:
        finput = self.im
        finput.seek(0)
        if finput.read(PNG._PNG_SIGNATURE_LENGTH) != PNG._PNG_SIGNATURE:
            return PNG._ERROR["wrong signature"]
        if PNG.freadBE32(finput) != PNG._PNG_IHDR_LENGTH:
            return PNG._ERROR["wrong header"]
        chunkType = finput.read(4)
        if chunkType != b"IHDR":
            return PNG._ERROR["wrong header"]
        header = finput.read(PNG._PNG_IHDR_LENGTH)
        self.IHDR["width"] = PNG.readBE32(header[:4])
        self.IHDR["height"] = PNG.readBE32(header[4:8])
        self.IHDR["bitdepth"] = int(header[8])
        self.IHDR["colortype"] = int(header[9])
        self.IHDR["compress"] = int(header[10])
        self.IHDR["filter"] = int(header[11])
        self.IHDR["interlace"] = int(header[12])
        crc = PNG.freadBE32(finput)
        checkCrc =zlib.crc32(chunkType + header)
        if not crc == checkCrc:
            return  PNG._ERROR["corrupted data"]
        while True:          
            chunkHead = finput.read(4)
            if len(chunkHead) == 0:
                break
            self.offset.append(finput.tell()-4)
            chunkLength = PNG.readBE32(chunkHead)
            chunkType = finput.read(4)
            self.topology.append(chunkType.decode("utf-8"))
            chunkData = finput.read(chunkLength)
            crc = PNG.freadBE32(finput)
            checkCrc =zlib.crc32(chunkType + chunkData)
            if not crc == checkCrc:
                return  PNG._ERROR["corrupted data"]
        return 0
    
    def getChunkData(self, chunkType:str)->bytes:
        data = b""
        finput = self.im
        for type, offset in zip(self.topology, self.offset):
            if chunkType == type:
                finput.seek(offset, 0)
                chunkLength =  PNG.freadBE32(finput)
                finput.read(4)
                data += finput.read(chunkLength)
        return data
    
    def unzipData(self) ->bytes:
        stream = self.getChunkData("IDAT")
        return zlib.decompress(stream)
    
    def _getChannelNb(self) -> int:
        colortype = self.IHDR["colortype"]
        has_tRNS = "tRNS" in self.topology
        use_palette = colortype & 1       #first bit
        is_truecolor = (colortype>>1) & 1 #second bit
        use_alpha = (colortype>>2) & 1    #third bit
        channelNb = 1
        if use_palette:
            channelNb += 2
            if has_tRNS:
                channelNb += 1
        else:
            if is_truecolor:
                channelNb += 2
            if use_alpha:
                channelNb += 1
        return channelNb
    
    def _getPixelbyteNb(self):
        bitdepth  = self.IHDR["bitdepth"]
        sampleByteNb = 1 if bitdepth <=8 else 2
        return sampleByteNb*self.bands

    def _toArray_RGBA(self, unzipStream) -> np.array:
        pixel_dtype = np.uint8 if self.IHDR["bitdepth"] <= 8 else np.uint16
        pre_data = np.frombuffer(unzipStream, dtype = pixel_dtype)
        IHDR = self.IHDR[0]
        height = IHDR["height"]
        width = IHDR["width"]
        #(height, scanlinewidth, bands)
        pre_array = pre_data.reshape(height, -1)
        _, scanlineWidth = pre_array.shape
        assert scanlineWidth == width*self.bands + 1
        data = pre_array[:,1:].reshape(height, -1, self.bands)
        filter_list = pre_array[:,0].reshape(-1)
        img = np.zeros_like(data)  
        for j in range(self.bands):
            print("bands", j)
            #deal with first line
            src = data[0,:,j]
            context = np.zeros_like(src)
            #the first byte of each scanline is the filter type
            filter_func = PNG_decoder.filter_dict[filter_list[0]]
            filter_func(img[0,:,j], context, src)
            for idx in range(1, height):
                src = data[idx,:,j]
                context = img[idx - 1,:,j]
                filter_func = PNG_decoder.filter_dict[filter_list[idx]]
                filter_func(img[idx,:,j], context, src)
        print(img.shape)
        return img
         
    def _toArray_PLTE(self, unzipStream):
        pass

    
if __name__ == "__main__":
    import pathlib
    import os
    from PIL import Image
    CUR_PATH = pathlib.Path(__file__).parent.resolve()
    file = os.path.join(CUR_PATH, "fish.png")
    with open(file, 'rb') as finput:
        axa = PNG_decoder(finput)
        #Todo: finish and wrap other possible modes in one toArray() function
        raw = axa._toArray_RGBA(axa.unzipData())
        im = Image.fromarray(raw, mode = "RGBA")
        outfile = os.path.join(CUR_PATH, "fish_cb.png")
        im.save(outfile)
        

        