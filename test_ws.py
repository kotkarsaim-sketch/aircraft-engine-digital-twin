import asyncio, websockets, json

async def test():
    async with websockets.connect('ws://localhost:8765') as ws:
        for i in range(3):
            msg = await ws.recv()
            data = json.loads(msg)
            print("Snapshot {}:".format(i+1))
            print("  mode: {}".format(data.get("mode")))
            print("  sensors: {}".format(list(data.get("sensors", {}).keys())))
            print("  ml_fault: {}".format(data.get("ml_fault")))
            print("  ml_confidence: {}".format(data.get("ml_confidence")))
            print("  ml_rul_h: {}".format(data.get("ml_rul_h")))
            probs = data.get("ml_probabilities")
            if probs:
                print("  ml_probabilities: {}".format({k: round(v, 3) for k, v in probs.items()}))

asyncio.run(test())