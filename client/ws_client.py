# -*- coding: utf-8 -*-
import websocket
import thread
import threading
import re
import time
import math
import solver
import config


startTurnTime = 0

lock = threading.Lock()

r_render = None
def AccessVarMSG(isRead, msg):
    res = None
    lock.acquire()
    if isRead:
        config.newMSG = False
        res = config.inputMsg
    else:
        config.inputMsg = msg
        config.newMSG = True
    lock.release()
    return res

def AccessVarResult(isRead, msg):
    res = None
    lock.acquire()
    if isRead:
        config.newRes = False
        res = config.solvedRes
    else:
        config.solvedRes = msg
        config.newRes = True
    lock.release()
    return res

def on_message(ws, message):
    global startTurnTime
    m_utf8 = message.decode('utf-8')
    print "->"
    if startTurnTime != 0:
        print "WARNING: time exceeded"
    startTurnTime = int(round(time.time()*1000))
    if True:
        pattern = u'^board=(.*)$'
        prog = re.compile(pattern)
        res = prog.match(m_utf8)
        parsed = res.groups()[0]
        #print len(parsed)
        sqrt_len = int(math.sqrt(len(parsed)))
        #print sqrt_len
        AccessVarMSG(False, parsed)
        #print config.newMSG
        #AccessVar(False, parsed)
        #print u"Recived message:\n\n" + res.groups()[0]
        #for i in range(len(parsed)):
        #    f.write(parsed[i])
        #    if i % sqrt_len == 0:
        #        f.write('\n')
        #f.write(parsed)
        #f.write('\n\n\n\n\n\n')
    

def on_error(ws, error):
    print error


def on_close(ws):
    print "### closed ###"


def on_open(ws):
    def run(*args):
        global startTurnTime
        while True:
            time.sleep(0.01)
            if config.newRes:
                print "<-"
                res = AccessVarResult(True, None)
                ws.send(res)
                currTime = int(round(time.time()*1000))
                print "time= " + str(currTime-startTurnTime)
                startTurnTime = 0

    thread.start_new_thread(run, ())


def startRenderThread():
    global r_render
    r_render = solver.Render()
    r_render.start()

if __name__ == "__main__":
    thread.start_new_thread(startRenderThread, ())
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("ws://tetrisj.jvmhost.net:12270/codenjoy-contest/ws?user=m16a",
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()
