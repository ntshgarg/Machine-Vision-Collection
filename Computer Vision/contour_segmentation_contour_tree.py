
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:48:36 2022

@author: nitishg
"""
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
sys.setrecursionlimit(4000)
class Contour:
    def __init__(self, contour):
        self.contour = contour
        self.prev = None
        self.next = None
      
class ContourDetection:
    def __init__(self,frame, level, segments):
        self.frame = frame
        self.level = level
        self.segments = segments
        self.visited = None
        self.segImg = None
        self.Graph = dict()
        self.SegmentCoordinates = dict()
        self.row = ( -1, -1, -1, 0, 1, 0, 1, 1 )
        self.col = ( -1, 1, 0, -1, -1, 1, 0, 1 )
        
    def _get_fraction(self, from_value,to_value):
        if (to_value == from_value):
            return 0
        return ((self.level - from_value) / (to_value - from_value))
    
    def _assemble_contours(self):
        current_index = 0
        contours = {}
        starts = {}
        ends = {}
        for from_point, to_point in self.segments:
            # Ignore degenerate segments.
            # This happens when (and only when) one vertex of the square is
            # exactly the contour level, and the rest are above or below.
            # This degenerate vertex will be picked up later by neighboring
            # squares.
            if from_point == to_point:
                continue
    
            tail, tail_num = starts.pop(to_point, (None, None))
            head, head_num = ends.pop(from_point, (None, None))
    
            if tail is not None and head is not None:
                # We need to connect these two contours.
                if tail is head:
                    # We need to closed a contour: add the end point
                    head.append(to_point)
                else:  # tail is not head
                    # We need to join two distinct contours.
                    # We want to keep the first contour segment created, so that
                    # the final contours are ordered left->right, top->bottom.
                    if tail_num > head_num:
                        # tail was created second. Append tail to head.
                        head.extend(tail)
                        # Remove tail from the detected contours
                        contours.pop(tail_num, None)
                        # Update starts and ends
                        starts[head[0]] = (head, head_num)
                        ends[head[-1]] = (head, head_num)
                    else:  # tail_num <= head_num
                        # head was created second. Prepend head to tail.
                        tail.extendleft(reversed(head))
                        # Remove head from the detected contours
                        starts.pop(head[0], None)  # head[0] can be == to_point!
                        contours.pop(head_num, None)
                        # Update starts and ends
                        starts[tail[0]] = (tail, tail_num)
                        ends[tail[-1]] = (tail, tail_num)
            elif tail is None and head is None:
                # We need to add a new contour
                new_contour = deque((from_point, to_point))
                contours[current_index] = new_contour
                starts[from_point] = (new_contour, current_index)
                ends[to_point] = (new_contour, current_index)
                current_index += 1
            elif head is None:  # tail is not None
                # tail first element is to_point: the new segment should be
                # prepended.
                tail.appendleft(from_point)
                # Update starts
                starts[from_point] = (tail, tail_num)
            else:  # tail is None and head is not None:
                # head last element is from_point: the new segment should be
                # appended
                head.append(to_point)
                # Update ends
                ends[to_point] = (head, head_num)
    
        return [np.array(contour) for _, contour in sorted(contours.items())]
    
    def _get_segments(self):
        for r0 in range(self.frame.shape[0] - 1):
            for c0 in range(self.frame.shape[1] - 1):
                r1, c1 = r0 + 1, c0 + 1
                
                #no masking req
                ul = self.frame[r0, c0].astype('float64')
                ur = self.frame[r0, c1].astype('float64')
                ll = self.frame[r1, c0].astype('float64')
                lr = self.frame[r1, c1].astype('float64')
                
                square_case = 0
                
                if ul > 0: square_case += 1
                if ur > 0: square_case += 2
                if ll > 0: square_case += 4
                if lr > 0: square_case += 8
                
                if square_case in [0,15]:
                    continue
                
                top = r0, c0 + self._get_fraction(ul, ur)
                bottom = r1, c0 + self._get_fraction(ll, lr)
                left = r0 + self._get_fraction(ul, ll), c0
                right = r0 + self._get_fraction(ur, lr), c1
            
                if (square_case == 1):
                    # top to left
                    self.segments.append((top, left))
                elif (square_case == 2):
                    # right to top
                    self.segments.append((right, top))
                elif (square_case == 3):
                    # right to left
                    self.segments.append((right, left))
                elif (square_case == 4):
                    # left to bottom
                    self.segments.append((left, bottom))
                elif (square_case == 5):
                    # top to bottom
                    self.segments.append((top, bottom))
                elif (square_case == 6):
                    self.segments.append((left, top))
                    self.segments.append((right, bottom))
                elif (square_case == 7):
                    # right to bottom
                    self.segments.append((right, bottom))
                elif (square_case == 8):
                    # bottom to right
                    self.segments.append((bottom, right))
                elif (square_case == 9):
                    self.segments.append((top, right))
                    self.segments.append((bottom, left))
                elif (square_case == 10):
                    # bottom to top
                    self.segments.append((bottom, top))
                elif (square_case == 11):
                    # bottom to left
                    self.segments.append((bottom, left))
                elif (square_case == 12):
                    # lef to right
                    self.segments.append((left, right))
                elif (square_case == 13):
                    # top to right
                    self.segments.append((top, right))
                elif (square_case == 14):
                    # left to top
                    self.segments.append((left, top))
    
    def _calculateSegmentArea(self, KeyNode):
        KeyCoordinates = self.SegmentCoordinates[KeyNode]
        Area = (KeyCoordinates[2] - KeyCoordinates[0]) * (KeyCoordinates[3] - KeyCoordinates[1])
        return Area
    
    def _establishParentChildRelationship(self):
        GraphCopy = self.Graph.copy()
        for key in self.Graph.keys():
            ChildNodes = self.Graph[key]
            ChildNodesCopy = ChildNodes.copy()
            KeyArea = self._calculateSegmentArea(key)
            for ChildNode in ChildNodesCopy:
                ChildArea = self._calculateSegmentArea(ChildNode)
                if ChildArea > KeyArea : 
                    ChildNodes.remove(ChildNode)
            if len(ChildNodes) == 0:
                del GraphCopy[key]
            self.Graph.update({key : ChildNodes})
            
        self.Graph = GraphCopy.copy()
                
    def _IsSafe(self, x, y):
        if x >= 0 and x < self.frame.shape[0] and y >= 0 and y < self.frame.shape[1]:
            return True
        return False
    
    def _showSegmentImage(self):
        plt.imshow(self.segImg)
        plt.show()
        
    def _getSegmentsAndNeighboursUtil(self, x, y, xp, yp):
        if not self._IsSafe(x, y) :
            return
        
        if self.visited[x,y]:
            if self.segImg[x,y] != self.segImg[xp,yp]:
                ParentSegmentLevel = self.segImg[xp,yp]
                ChildSegmentLevel = self.segImg[x,y]
                if ParentSegmentLevel not in self.Graph:
                    self.Graph.update({ParentSegmentLevel: [ChildSegmentLevel]})
                else:
                    ChildNodes = self.Graph[ParentSegmentLevel]
                    if ChildNodes.count(ChildSegmentLevel) > 0:
                        return
                    ChildNodes.append(ChildSegmentLevel)
                    self.Graph.update({ParentSegmentLevel: ChildNodes})
                    
                if ChildSegmentLevel not in self.Graph:
                    self.Graph.update({ChildSegmentLevel: [ParentSegmentLevel]})
                else:
                    ParentNodes = self.Graph[ChildSegmentLevel]
                    if ParentNodes.count(ParentSegmentLevel) > 0:
                        return
                    ParentNodes.append(ParentSegmentLevel)
                    self.Graph.update({ChildSegmentLevel: ParentNodes})
            return
        
        if self.frame[x,y] == self.frame[xp,yp]:
            self.visited[x,y] = True
            self.segImg[x,y] = self.segImg[xp,yp]
            ListOfCoordinates = self.SegmentCoordinates[int(self.segImg[x,y])]
            ListOfCoordinates[0] = min(x, ListOfCoordinates[0])
            ListOfCoordinates[1] = min(x, ListOfCoordinates[1])
            ListOfCoordinates[2] = max(y, ListOfCoordinates[2])
            ListOfCoordinates[3] = max(y, ListOfCoordinates[3])
            self._getSegmentsAndNeighboursUtil(x-1, y-1, x, y)
            self._getSegmentsAndNeighboursUtil(x-1, y  , x, y)
            self._getSegmentsAndNeighboursUtil(x-1, y+1, x, y)
            self._getSegmentsAndNeighboursUtil(x  , y-1, x, y)
            self._getSegmentsAndNeighboursUtil(x  , y+1, x, y)
            self._getSegmentsAndNeighboursUtil(x+1, y-1, x, y)
            self._getSegmentsAndNeighboursUtil(x+1, y  , x, y)
            self._getSegmentsAndNeighboursUtil(x+1, y+1, x, y)
    
    def _getSegmentsAndNeighbours(self):
        self.segImg = np.ones(self.frame.shape) * -1
        self.visited = np.zeros(self.frame.shape, dtype='bool')
        SegmentLevel = 0
        for i in range(self.frame.shape[0] - 1):
            for j in range(self.frame.shape[1] - 1):
                
                if not self.visited[i,j]:
                    self.visited[i,j] = True
                    self.segImg[i,j] = SegmentLevel
                    self.SegmentCoordinates.update({SegmentLevel : [i, j, i , j]})
                    SegmentLevel += 1
                    
                self._getSegmentsAndNeighboursUtil(i-1, j-1, i, j)
                self._getSegmentsAndNeighboursUtil(i-1, j  , i, j)
                self._getSegmentsAndNeighboursUtil(i-1, j+1, i, j)
                self._getSegmentsAndNeighboursUtil(i  , j-1, i, j)
                self._getSegmentsAndNeighboursUtil(i  , j+1, i, j)
                self._getSegmentsAndNeighboursUtil(i+1, j-1, i, j)
                self._getSegmentsAndNeighboursUtil(i+1, j  , i, j)
                self._getSegmentsAndNeighboursUtil(i+1, j+1, i, j)
    
    def _getSegmentsAndNeighboursBFSUtil(self, x, y):
        queue = []
        queue.append((x, y))
        self.visited[x, y] = True
        
        while len(queue) > 0:
            curr = queue.pop(0)
            
            for i in range(8):
                new_curr = (curr[0]+self.row[i], curr[1]+self.col[i])
                if self._IsSafe(new_curr[0], new_curr[1]):
                    
                    if self.visited[new_curr[0], new_curr[1]]:
                        if self.segImg[new_curr[0], new_curr[1]] != self.segImg[curr[0], curr[1]]:
                            ParentSegmentLevel = self.segImg[curr[0], curr[1]]
                            ChildSegmentLevel = self.segImg[new_curr[0], new_curr[1]]
                            if ParentSegmentLevel not in self.Graph:
                                self.Graph.update({ParentSegmentLevel: [ChildSegmentLevel]})
                            else:
                                ChildNodes = self.Graph[ParentSegmentLevel]
                                if ChildNodes.count(ChildSegmentLevel) > 0:
                                    continue
                                ChildNodes.append(ChildSegmentLevel)
                                self.Graph.update({ParentSegmentLevel: ChildNodes})
                                
                            if ChildSegmentLevel not in self.Graph:
                                self.Graph.update({ChildSegmentLevel: [ParentSegmentLevel]})
                            else:
                                ParentNodes = self.Graph[ChildSegmentLevel]
                                if ParentNodes.count(ParentSegmentLevel) > 0:
                                    continue
                                ParentNodes.append(ParentSegmentLevel)
                                self.Graph.update({ChildSegmentLevel: ParentNodes})
                        continue
                    
                    if self.frame[curr[0], curr[1]] == self.frame[new_curr[0],new_curr[1]]:
                        self.visited[new_curr[0], new_curr[1]] = True
                        self.segImg[new_curr[0], new_curr[1]] = self.segImg[curr[0], curr[1]]
                        ListOfCoordinates = self.SegmentCoordinates[int(self.segImg[new_curr[0],new_curr[1]])]
                        ListOfCoordinates[0] = min(new_curr[0], ListOfCoordinates[0])
                        ListOfCoordinates[1] = min(new_curr[0], ListOfCoordinates[1])
                        ListOfCoordinates[2] = max(new_curr[1], ListOfCoordinates[2])
                        ListOfCoordinates[3] = max(new_curr[1], ListOfCoordinates[3])
                        queue.append(new_curr)
    
    def _getSegmentsAndNeighboursBFS(self):
        self.segImg = np.ones(self.frame.shape) * -1
        self.visited = np.zeros(self.frame.shape, dtype='bool')
        SegmentLevel = 0
        
        for i in range(self.frame.shape[0]):
            for j in range(self.frame.shape[1]):
                if not self.visited[i,j]:
                    self.segImg[i,j] = SegmentLevel
                    
                    self.SegmentCoordinates.update({SegmentLevel : [i, j, i , j]})
                    SegmentLevel += 10
                    self._getSegmentsAndNeighboursBFSUtil(i, j)
                    
                    
        
if __name__ == '__main__':
    frame = cv2.imread(r"C:\Users\Nitish Garg\Desktop\dott.png", 0)
    frame = cv2.imread(r"C:\Users\Nitish Garg\Desktop\circularqrcode.png", 0)
    plt.imshow(frame)
    plt.show()
    #frame = cv2.imread(r"D:\OneDrive - Interra Systems Pvt Ltd\Desktop\CC60557H_QR_CODE_CNBC.png", 0)
    #frame = cv2.imread(r"C:\Users\Nitish Garg\Desktop\tiltedqrcode.jpg", 0)
    #frame = cv2.imread(r"C:\Users\Nitish Garg\Desktop\circularqrcode.png",0)
    #frame = cv2.imread(r"C:\Users\Nitish Garg\Desktop\dott.png",0)
    frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
    ret,frame = cv2.threshold(frame,127,255,cv2.THRESH_BINARY)
    
# =============================================================================
#     frame = np.zeros((50,50))
#     frame[10:20,10:20] = 255
#     frame[30:40,30:40] = 255
# =============================================================================
    level = 0.8
    segments = list()
    
    ContourDetectionInstance = ContourDetection(frame, level, segments)
# =============================================================================
#     ContourDetectionInstance._get_segments()
#     contours = ContourDetectionInstance._assemble_contours()
# =============================================================================
    
    ContourDetectionInstance._getSegmentsAndNeighboursBFS()
    ContourDetectionInstance._showSegmentImage()
    ContourDetectionInstance._establishParentChildRelationship()
