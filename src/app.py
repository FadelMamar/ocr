import litserve as ls
# from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock
import os
# from llama_index.llms.google_genai import GoogleGenAI
import base64
import logging
from ocr import GeminiExtractor, DspyExtractor, DoclingExtractor #, Extractor

ls.configure_logging(use_rich=True)

logger = logging.getLogger("API")

class OCR(ls.LitAPI):

    def setup(self, device):
        
        model = os.environ.get("MODEL")
        tmp = float(os.environ.get("TEMPERATURE",0.1))
        
        if os.environ("EXTRACTOR") == "dspy":
           self.extractor = DspyExtractor(model=model,
                                          temperature=tmp)
           
        elif os.environ("EXTRACTOR") == "gemini":
            self.extractor = GeminiExtractor(model=model,
                                             temperature=tmp)
        
        else:
            raise NotImplementedError()

    def decode_request(self, request):
        image = request.get('image',None)
        prompt = request.get('prompt',None)

        if image:
            try:
                image = base64.b64decode(image)
            except Exception as e:
                print(e)

        else:
            raise ValueError("Image is not provided")

        return dict(image=image,prompt=prompt) 
    
    def predict(self, x:dict):
        
        out = self.extractor.run(**x)
        
        return out

    def encode_response(self, output):
        return {"output": output} 

if __name__ == "__main__":
    api = OCR()
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=4242,pretty_logs=True,generate_client_file=False)