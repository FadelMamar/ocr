import litserve as ls
from llama_index.core.llms import ChatMessage, ImageBlock,TextBlock
import os
from llama_index.llms.google_genai import GoogleGenAI
import base64

ls.configure_logging(use_rich=True)

def perform_ocr(prompt:str,image:bytes, llm):

    msg = ChatMessage(
        role="user",
        blocks=[
            ImageBlock(image=image),
            TextBlock(text=prompt),
        ],
    )

    resp = llm.chat([msg])

    text = resp.message.blocks[0].text

    return text

PROMPT = "Extract the text from the image. Reply directly. DO NOT TRANSLATE"

class OCR(ls.LitAPI):

    def setup(self, device):

        api_key = os.environ.get("GOOGLE_API_KEY",None)

        if api_key is None:
            raise ValueError("GOOGLE_API_KEY env variable is not set.")
        
        model = os.environ.get("GOOGLE_MODEL","gemini-2.5-flash-preview-04-17")

        self.llm = GoogleGenAI(
            model=model,
            temperature=0.1
        )

    def decode_request(self, request):
        image = request.get('image',None)
        prompt = request.get('prompt',PROMPT)

        if image:
            try:
                image = base64.b64decode(image)
            except Exception as e:
                print(e)

        else:
            raise ValueError("Image is not provided")

        return dict(image=image,prompt=prompt) 
    
    def predict(self, x:dict):
        out = perform_ocr(llm=self.llm,**x)        
        return out

    def encode_response(self, output):
        return {"output": output} 

if __name__ == "__main__":
    port = 4242
    api = OCR()
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=port,pretty_logs=True)