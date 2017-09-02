#!/usr/bin/env python
#-*-coding:utf-8-*-


import wx
from numpy import *
import sitecustomize
import knn

class PaintWindow(wx.Window):
        def __init__(self, parent, id):
            wx.Window.__init__(self, parent, id)
            #self.SetBackgroundColour("Red")
            self.color = "Green"
            self.thickness = 3
        
            #创建一个画笔
            self.pen = wx.Pen(self.color, self.thickness, wx.SOLID)
            self.lines = []
            self.curLine = []
            self.pos = (0, 0)
            self.InitBuffer()
            
            #绘制的数据
            self.testData = [[0 for col in range(32)] for row in range(32)]
        
            #连接事件
            self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
            self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
            self.Bind(wx.EVT_MOTION, self.OnMotion)
            self.Bind(wx.EVT_SIZE, self.OnSize)
            self.Bind(wx.EVT_IDLE, self.OnIdle)
            self.Bind(wx.EVT_PAINT, self.OnPaint)
            
            panel = wx.Panel(self,size=(320,130),pos=(0,290))
            panel.SetBackgroundColour("gray")
            button1 = wx.Button(panel,label="清空".decode('UTF-8'),pos=(60,10),size=(60,30))
            button2 = wx.Button(panel,label="识别".decode('UTF-8'),pos=(170,10),size=(60,30))
            self.Bind(wx.EVT_BUTTON,self.OnClearDraw,button1)
            self.Bind(wx.EVT_BUTTON,self.OnRecogniton,button2)
            self.posCtrl=wx.TextCtrl(panel,-1,'',pos=(10,50),size=(300,20),style=wx.TE_LEFT)#创建文本框控件  
            klabel=wx.StaticText(panel,-1,"最邻近点个数(k值):".decode('UTF-8'),pos=(10,80))
            self.posCtrlk=wx.TextCtrl(panel,-1,'',pos=(120,78),size=(40,20),style=wx.TE_LEFT)#创建文本框控件  
            self.posCtrlk.SetValue('5')
            
            klabel=wx.StaticText(panel,-1,"输入正确的值并更正:".decode('UTF-8'),pos=(10,105))
            self.posCtrlck=wx.TextCtrl(panel,-1,'',pos=(125,103),size=(40,20),style=wx.TE_LEFT)#创建文本框控件  
            button3 = wx.Button(panel,label="更正".decode('UTF-8'),pos=(175,103),size=(40,20))
            self.Bind(wx.EVT_BUTTON,self.OnCorrect,button3)
            #初始化训练样本  
            self.trainingMat,self.hwLabels,self.classNumDic = knn.getTrainSample()
            
        def OnCorrect(self,event):
            correctClass = int(self.posCtrlck.GetValue())
            filename="trainingDigits/"+str(correctClass)+'_'+str(self.classNumDic[correctClass]+1)+".added.txt"
            file_object = open(filename, 'w')
            for i in range(32):
                for j in range(32):
                    file_object.write(str(self.testData[i][j]))
                file_object.write("\n")
            file_object.close( )
            self.trainingMat,self.hwLabels,self.classNumDic = knn.getTrainSample()
            mDlg = wx.MessageBox('入库完成!'.decode('UTF-8'))
            
        def OnClearDraw(self,event):
            self.lines = []
            self.testData = [[0 for col in range(32)] for row in range(32)]
            self.InitBuffer()
            self.Refresh()
        def OnRecogniton(self,event):
            testVector = zeros((1,1024))
            for i in range(32):
                for j in range(32):
                    testVector[0,32*i+j] = self.testData[i][j]
                    
            kvalue = int(self.posCtrlk.GetValue())
            re = knn.handwritingClassifyValidate(testVector,self.trainingMat,self.hwLabels,kvalue)
            self.posCtrl.AppendText(str(re))
        def InitBuffer(self):
            size = self.GetClientSize()
            
            #创建缓存的设备上下文
            self.buffer = wx.EmptyBitmap(size.width+5, size.height-100)
            dc = wx.BufferedDC(None, self.buffer)
            
            #使用设备上下文
            #dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
            dc.SetBackground(wx.Brush("red"))
            dc.Clear()
            self.DrawLines(dc)
            self.reInitBuffer = False
            
        def GetLinesData(self):
            return self.lines[:]
        
        def SetLinesData(self, lines):
            self.lines = lines[:]
            self.InitBuffer()
            self.Refresh()
            
        def OnLeftDown(self, event):
            self.curLine = []
            
            #获取鼠标位置
            self.pos = event.GetPositionTuple()
            self.CaptureMouse()
            
        def OnLeftUp(self, event):
            if self.HasCapture():
                self.lines.append((self.color,
                                   self.thickness,
                                   self.curLine))
                self.curLine = []
                self.ReleaseMouse()
                
        def OnMotion(self, event):
            if event.Dragging() and event.LeftIsDown():
                dc = wx.BufferedDC(wx.ClientDC(self), self.buffer)
                self.drawMotion(dc, event)
            event.Skip()
        
        def drawMotion(self, dc, event):
            dc.SetPen(self.pen)
            newPos = event.GetPositionTuple()
            tx,ty=newPos
            ntx,nty=int(tx/10),int(ty/10)
            self.testData[nty][ntx]=1
            radius = int(self.thickness/2)
            for i in range(-radius,radius+1):
                for j in range(-radius,radius+1):
                    zntx,znty=ntx+i,nty+j
                    if (zntx>31):zntx=31
                    if (zntx<0):zntx=0
                    if (znty>31):zntx=31
                    if (znty<0):zntx=0
                    self.testData[znty][zntx]=1
            coords = self.pos + newPos
            self.curLine.append(coords)
            dc.DrawLine(*coords)
            self.pos = newPos
            
        def OnSize(self, event):
            self.reInitBuffer = True
        
        def OnIdle(self, event):
            if self.reInitBuffer:
                self.InitBuffer()
                self.Refresh(False)
        
        def OnPaint(self, event):
            dc = wx.BufferedPaintDC(self, self.buffer)
            
        def DrawLines(self, dc):
            for colour, thickness, line in self.lines:
                pen = wx.Pen(colour, thickness, wx.SOLID)
                dc.SetPen(pen)
                for coords in line:
                    dc.DrawLine(*coords)
        
        def SetColor(self, color):
            self.color = color
            self.pen = wx.Pen(self.color, self.thickness, wx.SOLID)
            
        def SetThickness(self, num):
            self.thickness = num
            self.pen = wx.Pen(self.color, self.thickness, wx.SOLID)
            
class PaintFrame(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, -1, "手写数字识别演示程序".decode('UTF-8'),pos=(400,100),style=wx.DEFAULT_FRAME_STYLE^(wx.RESIZE_BORDER|wx.MINIMIZE_BOX|wx.MAXIMIZE_BOX),
                           size = (325, 450))
        self.paint = PaintWindow(self, -1)

class knnApp(wx.App):
    def OnInit(self):
        bmp = wx.Image("splash.jpg").ConvertToBitmap()  
        wx.SplashScreen(bmp,wx.SPLASH_CENTER_ON_SCREEN | wx.SPLASH_TIMEOUT,3,None,-1)  
        wx.Yield() 
        frame = PaintFrame(None)
        frame.Show(True)
        return True
        
app = knnApp()
app.MainLoop()