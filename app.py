import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import tk_tools
import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.svm
import sklearn.datasets


#====================================
# Window Setting & Layout
#====================================
root = tk.Tk()
root.title("GDS ToolKit")
# root.config(bg="skyblue")
# root.geometry("550x300+300+150")
root.resizable(width=True, height=True)

upFrame = tk.Frame(root, width=800, height=50)
upFrame.grid(row=0, column=0, padx=10, pady=5)

leftFrame = tk.Frame(root, width=400, height=400)
leftFrame.grid(row=1, column=0, padx=10, pady=5)

rightFrame = tk.Frame(root, width=400, height=400)
rightFrame.grid(row=1, column=1, padx=10, pady=5)

#====================================
# OptionMenu
#====================================
def SmartOptionMenuCallback():
    if optionMenu.get()=='QRCode Decoder':
        for widget in leftFrame.winfo_children():
            widget.grid_forget()
        for widget in rightFrame.winfo_children():
            widget.grid_forget()
        QROpenImageBtn.grid(row=1, column=0, padx=5, pady=5)
        QRDecoderBtn.grid(row=1, column=0, padx=5, pady=5)
    elif optionMenu.get()=='Optimal Fiber Point':
        for widget in leftFrame.winfo_children():
            widget.grid_forget()
        for widget in rightFrame.winfo_children():
            widget.grid_forget()
        OptimLabelIter.grid(row=0, column=0, padx=5, pady=5)
        OptimEntryIter.grid(row=0, column=1, padx=5, pady=5)
        OptimLabelBound.grid(row=1, column=0, padx=5, pady=5)
        OptimEntryBound.grid(row=1, column=1, padx=5, pady=5)
        OptimBtn.grid(row=3, column=0, padx=5, pady=5)
        

optionMenu = tk_tools.SmartOptionMenu(upFrame, ['QRCode Decoder', 'Optimal Fiber Point'])
optionMenu.add_callback(SmartOptionMenuCallback)
optionMenu.grid(row=0, column=0, padx=5, pady=5)

#====================================
# QRCode Decoder
#====================================
from LabelQR.QRCode import decoderStandard

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename

def openImage():
    imgPath = openfn()
    img = Image.open(imgPath)
    QRDecoderBtnImagePath.set(imgPath)
    N, M = np.array(img).shape[:2]
    if (N>M):
        img = img.resize((M*250//N, 250))
    else:
        img = img.resize((250, N*250//M))
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(rightFrame, image=img)
    panel.image = img
    panel.grid()

def decoder():
    if QRDecoderBtnImagePath.get()=='':
        tk.Label(rightFrame, text="Please Open one Image First!").grid()
        return
    decodedText = decoderStandard.readQRCode(QRDecoderBtnImagePath.get())
    tk.Label(rightFrame, text=decodedText).grid()

QRDecoderBtnImagePath = tk.StringVar()
QROpenImageBtn = tk.Button(leftFrame, text='open image', command=openImage)
QRDecoderBtn   = tk.Button(rightFrame, text='Decode!', command=decoder)

#====================================
# Optimal Fiber Point
#====================================
from GD_Optim import Optimizer
import threading
import queue

def OptimSubmit():
    # tk.Label(rightFrame, text = OptimLabel.cget("text")+" "+OptimTmp.get()).grid()
    # OptimQuery.set(OptimTmp.get())
    tmp = queryAPI(result)
    tk.Label(rightFrame, text = OptimLabel.cget("text")+f" {tmp}").grid()
    OptimQuery.set(tmp)
    inputReady.set()

def queryAPI(list0):
    x, y = list0
    return np.exp( (-(x-1)**2 - (y-2)**2) / (2*(1**2)) )
#     # ret = float(input(f"Input Coordination {x,y} Value:"))
#     sample_loss.put(params)
#     block_query.wait(timeout=1)
#     while not end_program.is_set() and not block_query.is_set():
#         block_query.wait(timeout=1)
#     if end_program.is_set():
#         return None, None
#     y_list.append(float(block_query_var.get()))
#     block_query.clear()

def bayeOptimize():
    # Optimizer.bayesianOptimisation(sample_loss=queryAPI, bounds=np.array([[-10,10],[-10,10]]), n_iters=20)
    OptimLabel.grid(row=4,column=0)
    OptimEntry.grid(row=4,column=1)
    OptimSubmit.grid(row=4,column=3)
    OptimState.set(1)
    if OptimIter.get()=='':
        tk.Label(rightFrame, text="Please Enter Iterations First!").grid()
        return
    iterations = int(OptimIter.get())
    if OptimIter.get()=='':
        tk.Label(rightFrame, text="Please Enter Optim Bounds First!").grid()
        return
    tmpBounds = np.fromstring(OptimBound.get(),sep=',')
    bounds = []
    for i in range(0,tmpBounds.shape[0],2):
        bounds.append(tmpBounds[i:i+2])
    print(iterations, tmpBounds, bounds)
    global threadOptim
    threadOptim = threading.Thread(target=Optimizer.bayesianOptimisation,args=(iterations-5, queryPoints, inputReady, OptimQuery, endProgram, np.array(bounds)))
    threadOptim.daemon = True
    threadOptim.start()


def OptimRetrieveResult():
    if OptimState.get()==0:
        root.after(1000, OptimRetrieveResult) 
        return
    try:
        global result
        result = queryPoints.get_nowait()  
        if (result[0] == "Finish"):
            tk.Label(leftFrame, text = f"Optimization Finished! Optimal Position: {result[1]}, Optimal Value: {result[2]}").grid()
            OptimLabel.config(text=f"Input Coordination [Empty] Value:")
        else:
            OptimLabel.config(text=f"Input Coordination {result} Value:")
        print("Received from Function F:", result)
    except queue.Empty:
        print("No result available yet.")
    root.after(1000, OptimRetrieveResult) 

queryPoints = queue.Queue()
OptimState = tk.IntVar(root,0)
OptimTmp   = tk.StringVar()
OptimQuery = tk.StringVar()
OptimIter  = tk.StringVar()
OptimBound = tk.StringVar()
inputReady = threading.Event()
endProgram = threading.Event()

OptimLabelIter  = tk.Label(leftFrame, text = f"Input number of Iterations (>5):")
OptimEntryIter  = tk.Entry(leftFrame, textvariable = OptimIter)
OptimLabelBound = tk.Label(leftFrame, text = f"Input bounds (e.g. xLow, xHigh, yLow, yHigh):")
OptimEntryBound = tk.Entry(leftFrame, textvariable = OptimBound)
OptimBtn = tk.Button(leftFrame, text='Start Optimize!', command = bayeOptimize)

OptimLabel  = tk.Label(leftFrame, text = f"Input Coordination [Empty] Value:")
OptimEntry  = tk.Entry(leftFrame, textvariable = OptimTmp)
OptimSubmit = tk.Button(leftFrame, text = 'Confirm', command = OptimSubmit)

#====================================
# Run App
#====================================
def on_close():
    if tk.messagebox.askokcancel("Quit", "Do you want to quit?"):
        try:
            endProgram.set()
            if threadOptim.is_alive():
                threadOptim.join()
        except:
            print("Failed!")
            a = None
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.after(1000, OptimRetrieveResult)
root.mainloop()