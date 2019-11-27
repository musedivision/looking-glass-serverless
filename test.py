import base64

# from lambda_local.main import call
# from lambda_local.context import Context

import handler


    # res = call(handler.handler, event, context)

with open("test/lush.jpg", "rb") as f:
    data = f.read()
    img64 = base64.encodebytes(data)

    event = {
            "body":img64
            }
    
    res = handler.handler(event, None)
    print(res["body"])
