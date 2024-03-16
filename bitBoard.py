import chess
import numpy as np
from weights import WEIGHTS
from structEl import structEl
from math import floor

COLOR_OFFSET = {chess.WHITE: 0, chess.BLACK: 1}
BOOL_BIN = {True: 1, False: 0}

class bitBoard:

    def __init__(self,board=None,bruteForce = False):
        self.bF = bruteForce
        self.__list = np.asarray([None]*(768+5)) # private 773-bits bit array: 64*6*2 + 5 ~ 64 cases for each piece of each color plus side to move & castling rights
        self.__pieceswght = np.asarray([None]*(768+5))
        self.setWeights()
        self.strEl = structEl()
        if(board==None): # no board provided, initial board is supposed
            self.board = chess.Board()
        else:
            self.board = board   

        self.setList(self.board)
        self.wghtInit = np.sum(abs(self.getWeighted())) - 16
        self.__value = self.getEval()

    def value(self):
        return self.__value
    
    def getList(self):
        return self.__list
    
    def getWeighted(self, lst=None):
        if lst is None:
            return self.__list * self.__pieceswght
        else:
            return lst * self.__pieceswght

    def gamePhase(self):
        weighted = abs(self.getWeighted())
        weighted[0:64] = np.asarray(64*[0])
        weighted[6*64:7*64] = np.asarray(64*[0])
        weight = np.sum(weighted)
        return 1 - weight/self.wghtInit
    
    def getEval(self):
        phase = self.gamePhase() # goes from 0% to 100% depending on the weight of non-pawn material remaining
        a1 = 5/6 # material
        a2 = (1-a1)/7 # b ctl
        a3 = (1-a1)/7 # q ctl
        a4 = (1-a1)/7 # r ctl
        a5 = (1-a1)/7 # k ctl
        a6 = (1-a1)/10 # p ctl
        a7 = (1-a1)/10 # p pen
        a8 = (1 -a1)/7 # king exposed
        a9 = (1-a1)/14 # king protected
        if(self.bF == True):
            res = a1*np.sum(self.getWeighted()) + a2*self.bishopCtlBf() + a3*(0.5+0.5*phase)*self.queenCtlBf() + phase*a4*self.rookCtlBf() + (1+0.5*(1-phase))*a5*self.knightCtl() + a6*self.pawnCtl() - a7*self.pawnPenalty() + a9*self.kingProtected()
        else: 
            res = a1*np.sum(self.getWeighted()) + a2*self.bishopCtl() + a3*(0.5+0.5*phase)*self.queenCtl() + phase*a4*self.rookCtl() + (1+0.5*(1-phase))*a5*self.knightCtl() + a6*self.pawnCtl() - a7*self.pawnPenalty() + a9*self.kingProtected()
        self.__value = res
        return res
    
    def getDeltaEval(self, prevList, prevBboardValue):
        a1 = 5/6 # material
        a2 = (1-a1)/7 # b ctl
        a3 = (1-a1)/7 # q ctl
        a4 = (1-a1)/7 # r ctl
        a5 = (1-a1)/7 # k ctl
        a6 = (1-a1)/10 # p ctl
        a7 = (1-a1)/10 # p pen
        a8 = (1 -a1)/7 # king exposed
        a9 = (1-a1)/14 # king protected
        prevBboardList = prevList.copy()
        listDiff = self.__list - prevBboardList
        weightedDiff = self.getWeighted(listDiff)
        res = a1*np.sum(weightedDiff) 
        color = True
        indexes = []
        for i in range(len(listDiff)):
            if listDiff[i]!=0:
                if not indexes:
                    indexes.append(i)
                else:    
                    if(i-indexes[-1]>=64):
                        indexes.append(i)
                    else:
                        prevBboardList[indexes[-1]]=0
                        prevBboardList[i]=0        

        wdiffSponge = self.sponge(weightedDiff)
        prevSponge = self.sponge(prevBboardList)
        dirbis = [self.strEl.NE, self.strEl.NW, self.strEl.SE, self.strEl.SW]
        dirrk = [self.strEl.S, self.strEl.N, self.strEl.E, self.strEl.W]
        for i in range(len(wdiffSponge)):
            if(wdiffSponge[i]!=0):
                convkg = self.strEl.Kg[i]
                convkn =  self.strEl.Kn[i]
                convbp = self.strEl.wP[i]
                convwp = self.strEl.bP[i]
                convpp = self.strEl.Pp[i]
                convbis = np.asarray([0]*64)
                convrk = np.asarray([0]*64)
                convqn = np.asarray([0]*64)
                for d in dirbis:
                    j = 0
                    while(np.sum(d[-j][i]@prevSponge)>1):
                        j += 1
                    convbis += d[-j][i]
                for d in dirrk:
                    j = 0
                    while(np.sum(d[-j][i]@prevSponge)>1):
                        j += 1
                    convrk += d[-j][i]    
                convqn = convbis + convrk

                nwkg = convkg@prevBboardList[5*64:6*64]
                nbkg = convkg@prevBboardList[11*64:12*64]
                nwkn = convkn@prevBboardList[1*64:2*64]
                nbkn = convkn@prevBboardList[7*64:8*64]
                nwbs = convbis@prevBboardList[2*64:3*64]
                nbbs = convbis@prevBboardList[8*64:9*64]
                nwrk = convrk@prevBboardList[3*64:4*64]
                nbrk = convrk@prevBboardList[9*64:10*64]
                nwq = convqn@prevBboardList[4*64:5*64]
                nbq = convqn@prevBboardList[10*64:11*64]
                nwp = convwp@prevBboardList[0:64]
                nbp = convbp@prevBboardList[6*64:7*64]

                if(nwbs>=1):
                    maskedPrev = []
                    for i in range(12):
                        maskedPrev.extend(prevBboardList[i*64:(i+1)*64]*convbis)
                    maskedPrev.extend([0]*5)    
                    maskedPrev = np.asarray(maskedPrev)     
                    if(wdiffSponge[i]>0):
                        res -= a2*self.subBishopCtl(True, maskedPrev.copy(),prevBboardList[2*64:3*64]*convbis)
                    elif(wdiffSponge[i]<0):   
                        res += a2*self.subBishopCtl(True, maskedPrev.copy(),prevBboardList[2*64:3*64]*convbis)
                elif(nbbs>=1):
                    maskedPrev = []
                    for i in range(12):
                        maskedPrev.extend(prevBboardList[i*64:(i+1)*64]*convbis)
                    maskedPrev.extend([0]*5)    
                    maskedPrev = np.asarray(maskedPrev)         
                    if(wdiffSponge[i]>0):
                        res -= a2*self.subBishopCtl(False, maskedPrev.copy(),prevBboardList[8*64:9*64]*convbis)
                    elif(wdiffSponge[i]<0):   
                        res += a2*self.subBishopCtl(False, maskedPrev.copy(),prevBboardList[8*64:9*64]*convbis)
                elif(nwrk>=1):
                    maskedPrev = []
                    for i in range(12):
                        maskedPrev.extend(prevBboardList[i*64:(i+1)*64]*convrk)
                    maskedPrev.extend([0]*5)    
                    maskedPrev = np.asarray(maskedPrev)     
                    if(wdiffSponge[i]>0):
                        res -= a4*self.subRookCtl(True, maskedPrev.copy(),prevBboardList[3*64:4*64]*convrk)
                    elif(wdiffSponge[i]<0):   
                        res += a4*self.subRookCtl(True, maskedPrev.copy(),prevBboardList[3*64:4*64]*convrk)   
                elif(nbrk>=1):
                    maskedPrev = []
                    for i in range(12):
                        maskedPrev.extend(prevBboardList[i*64:(i+1)*64]*convrk) 
                    maskedPrev.extend([0]*5)        
                    maskedPrev = np.asarray(maskedPrev) 
                    if(wdiffSponge[i]>0):
                        res -= a4*self.subRookCtl(False, maskedPrev.copy(),prevBboardList[9*64:10*64]*convrk)
                    elif(wdiffSponge[i]<0):   
                        res += a4*self.subRookCtl(False, maskedPrev.copy(),prevBboardList[9*64:10*64]*convrk) 
                elif(nwq>=1):
                    maskedPrev = []
                    for i in range(12):
                        maskedPrev.extend(prevBboardList[i*64:(i+1)*64]*convqn)
                    maskedPrev.extend([0]*5)    
                    maskedPrev = np.asarray(maskedPrev)         
                    if(wdiffSponge[i]>0):
                        res -= a3*self.subQueenCtl(True, maskedPrev.copy(), prevBboardList[4*64:5*64]*convqn)
                    elif(wdiffSponge[i]<0):   
                        res += a3*self.subQueenCtl(True, maskedPrev.copy(), prevBboardList[4*64:5*64]*convqn) 
                elif(nbq>=1):  
                    maskedPrev = []
                    for i in range(12):
                        maskedPrev.extend(prevBboardList[i*64:(i+1)*64]*convqn)
                    maskedPrev.extend([0]*5)    
                    maskedPrev = np.asarray(maskedPrev)         
                    if(wdiffSponge[i]>0):
                        res -= a3*self.subQueenCtl(False, maskedPrev.copy(),prevBboardList[10*64:11*64]*convqn)
                    elif(wdiffSponge[i]<0):   
                        res += a3*self.subQueenCtl(False, maskedPrev.copy(),prevBboardList[10*64:11*64]*convqn)                   

                res += a9*(nwkg - nbkg) + wdiffSponge[i]*(a5*(nwkn - nbkn) + a2*(nwbs - nbbs) + a4*(nwrk - nbrk) + a3*(nwq - nbq) + a6*(nwp - nbp))

        for i in indexes:
            if(i/64>=6):
                color = False

            if(i/64<6):
                color = True    

            if(i in np.asarray(range(0,64))): # white pawn moved
                ctl = a6*self.subPawnCtl(color, prevBboardList, listDiff[0:64].copy()) 
                pen = - a7*self.subPawnPenalty(color,prevBboardList, listDiff[0:64].copy())
                res+=ctl + pen  
            

            elif(i in np.asarray(range(64,2*64))): # white knight moved 
                ctl = a5*self.subKnightCtl(color, prevBboardList.copy(), listDiff[64:2*64])
                res+=ctl    
            

            elif(i in np.asarray(range(2*64,3*64))): # white bishop moved    
                ctl = a2*self.subBishopCtl(color, prevBboardList.copy(), listDiff[2*64:3*64])
                res+=ctl    
            

            elif(i in np.asarray(range(3*64,4*64))): # white rook moved   
                ctl = a4*self.subRookCtl(color, prevBboardList.copy(), listDiff[3*64:4*64])
           

            elif(i in np.asarray(range(4*64,5*64))): # white queen moved
                ctl = a3*self.subQueenCtl(color, prevBboardList.copy(), listDiff[4*64:5*64])
                res+=ctl    
          

            elif(i in np.asarray(range(5*64,6*64))): # white king moved 
                ctl = a9*self.subKingProtected(color, prevBboardList.copy(), listDiff[5*64:6*64])
                res+=ctl      

            elif(i in np.asarray(range(6*64,7*64))): # black pawn moved    
                ctl = a6*self.subPawnCtl(color, prevBboardList.copy(), listDiff[6*64:7*64]) 
                pen = - a7*self.subPawnPenalty(color,prevBboardList.copy(), listDiff[6*64:7*64])
                res-=ctl + pen  

            elif(i in np.asarray(range(7*64,8*64))): # black knight moved    
                ctl = a5*self.subKnightCtl(color, prevBboardList.copy(), listDiff[7*64:8*64])
                res-=ctl    
            

            elif(i in np.asarray(range(8*64,9*64))): # black bishop moved    
                ctl = a2*self.subBishopCtl(color, prevBboardList.copy(), listDiff[8*64:9*64])
                res-=ctl  

            elif(i in np.asarray(range(9*64,10*64))): # black rook moved    
                ctl = a4*self.subRookCtl(color, prevBboardList.copy(), listDiff[9*64:10*64])
                res-=ctl  

            elif(i in np.asarray(range(10*64,11*64))): # black queen moved    
                ctl = a3*self.subQueenCtl(color, prevBboardList.copy(), listDiff[10*64:11*64])
                res-=ctl  

            elif(i in np.asarray(range(11*64,12*64))): # black king moved
                ctl = a9*self.subKingProtected(color, prevBboardList.copy(), listDiff[11*64:12*64])
                res-=ctl 
        
        self.__value = res + prevBboardValue
        return self.__value
    
    def pawnPenalty(self):
        return self.subPawnPenalty(True, self.__list.copy()) - self.subPawnPenalty(False, self.__list.copy())

    def kingProtected(self):
        return self.subKingProtected(True, self.__list.copy()) - self.subKingProtected(False, self.__list.copy())

    def pawnCtl(self):
        return self.subPawnCtl(True, self.__list.copy()) - self.subPawnCtl(False, self.__list.copy()) 

    def knightCtl(self):
        return self.subKnightCtl(True, self.__list.copy()) - self.subKnightCtl(False, self.__list.copy())

    def rookCtl(self):
        return self.subRookCtl(True, self.__list.copy()) - self.subRookCtl(False, self.__list.copy())
    
    def queenCtl(self):
        return self.subQueenCtl(True, self.__list.copy()) - self.subQueenCtl(False, self.__list.copy())
    
    def bishopCtl(self):
        return self.subBishopCtl(True, self.__list.copy()) - self.subBishopCtl(False, self.__list.copy())
    
    def rookCtlBf(self):
        return self.subRookCtlBf(True, self.__list.copy()) - self.subRookCtlBf(False, self.__list.copy())
    
    def queenCtlBf(self):
        return self.subQueenCtlBf(True, self.__list.copy()) - self.subQueenCtlBf(False, self.__list.copy())
    
    def bishopCtlBf(self):
        return self.subBishopCtlBf(True, self.__list.copy()) - self.subBishopCtlBf(False, self.__list.copy())
    
    def subKingProtected(self, color, lst, King=None):
        if(color==True):
            lst[6*64:12*64] = np.asarray([0]*6*64)
            if(King is None):
                King = lst[5*64:6*64]
        else:
            lst[0:6*64] = np.asarray([0]*6*64)
            if(King is None):
                King = lst[11*64:12*64]
        sponge = self.sponge(lst)
        conv = [0]*64
        for i in range(64):
            if(King[i]==1):
                conv += self.strEl.Kg[i]
            elif(King[i]==-1):
                conv -= self.strEl.Kg[i]    
        return sponge@conv    

    def subPawnCtl(self, color, lst, pawns = None):
        weighted = self.getWeighted(lst)
        if(color==True):
            advKing = lst[11*64:12*64]
            if(pawns is None):
                pawns = lst[0:64]
        else:   
            advKing = lst[5*64:6*64]
            if(pawns is None):
                pawns = lst[6*64:7*64] 

        sponge = abs(self.sponge(weighted)) + 9*advKing
        conv = np.asarray(64*[0])
        
        for i in range(64):
            if(pawns[i]==1 and color==True):
                conv += self.strEl.wP[i]
            elif(pawns[i]==1 and color==False):
                conv += self.strEl.bP[i]
            elif(pawns[i]==-1 and color==True):
                conv -= self.strEl.wP[i]
            elif(pawns[i]==-1 and color==False):
                conv -= self.strEl.bP[i]


        return sponge@conv

    def subPawnPenalty(self,color, lst, pawns = None):
        if(color==True):  
            sponge = lst[0:64]
            if(pawns is None):
                pawns=sponge
        else:
            sponge = lst[6*64:7*64]
            if(pawns is None):
                pawns=sponge
    
        conv = np.asarray(64*[0])

        for i in range(64):
            if(pawns[i]==1):
                conv += self.strEl.Pp[i]
            elif(pawns[i]==-1):
                conv -= self.strEl.Pp[i]    

        return sponge@conv 

    def subBishopCtlBf(self, color, lst, bishops = None):
        weighted = self.getWeighted(lst)
        if(color==True):
            advKing = lst[11*64:12*64]
            if(bishops is None):
                bishops = lst[2*64:3*64]
        else:
            advKing = lst[5*64:6*64]
            if(bishops is None):
                bishops = lst[8*64:9*64]
    
        sponge = abs(self.sponge(weighted)) + 9*advKing
        conv = np.asarray(64*[0])

        for i in range(64):
            if(bishops[i]==1):
                conv += self.strEl.Bp[i]
            elif(bishops[i]==-1):
                conv -= self.strEl.Bp[i]    
        
        return sponge@conv
    
    def subQueenCtlBf(self, color, lst, queen = None): 
        weighted = self.getWeighted(lst)
        if(color==True):
            advKing = lst[11*64:12*64]
            if(queen is None):
                queen = lst[4*64:5*64]
        else:    
            advKing = lst[5*64:6*64]
            if(queen is None):
                queen = lst[10*64:11*64]

        sponge = abs(self.sponge(weighted)) + 9*advKing

        conv = np.asarray(64*[0])
        
        for i in range(64):
            if(queen[i]==1):
                conv += self.strEl.Qn[i]
            elif(queen[i]==-1):
                conv -= self.strEl.Qn[i]

        return sponge@conv
    
    def subRookCtlBf(self, color, lst, rooks = None): 
        weighted = self.getWeighted(lst)
        if(color==True):
            advKing = lst[11*64:12*64]
            if(rooks is None):
                rooks = lst[3*64:4*64]
        else:
            advKing = lst[5*64:6*64]
            if(rooks is None):
                rooks = lst[9*64:10*64]
    
        sponge = abs(self.sponge(weighted)) + 9*advKing

        conv = np.asarray(64*[0])
        
        for i in range(64):
            if(rooks[i]==1):
                conv += self.strEl.Rk[i]
            elif(rooks[i]==-1):
                conv -= self.strEl.Rk[i]    
    
        return sponge@conv
    
    def subBishopCtl(self, color, lst, bishops = None):
        unweightspg = self.sponge(lst)
        weighted = self.getWeighted(lst)
        if(color==True):
            if(bishops is None):
                bishops = lst[2*64:3*64]
            advKing = lst[11*64:12*64]
        else:
            if(bishops is None):
                bishops = lst[8*64:9*64]
            advKing = lst[5*64:6*64]
        sponge = abs(self.sponge(weighted)) + 9*advKing
        conv = np.asarray(64*[0])
        dir = [self.strEl.NE, self.strEl.NW, self.strEl.SE, self.strEl.SW]
        for i in range(64):
            if(bishops[i]==1):
                for d in dir:
                    j = 0
                    while(np.sum(d[-j][i]@unweightspg)>1):
                        j += 1
                    conv += d[-j][i]
            elif(bishops[i]==-1):
                for d in dir:
                    j = 0
                    while(np.sum(d[-j][i]@unweightspg)>1):
                        j += 1
                    conv -= d[-j][i]        
        return sponge@conv
    
    def subQueenCtl(self, color, lst, queen = None): 
        unweightspg = self.sponge(lst)
        weighted = self.getWeighted(lst)
        if(color==True):
            if(queen is None):
                queen = lst[4*64:5*64]
            advKing = lst[11*64:12*64]
        else:
            if(queen is None):
                queen = lst[10*64:11*64]
            advKing = lst[5*64:6*64]
        sponge = abs(self.sponge(weighted)) + 9*advKing
        conv = np.asarray(64*[0])
        dir = [self.strEl.NE, self.strEl.NW, self.strEl.SE, self.strEl.SW,self.strEl.S, self.strEl.N, self.strEl.E, self.strEl.W]
        for i in range(64):
            if(queen[i]==1):
                for d in dir:
                    j = 0
                    while(np.sum(d[-j][i]@unweightspg)>1):
                        j += 1
                    conv += d[-j][i]
            elif(queen[i]==-1):
                for d in dir:
                    j = 0
                    while(np.sum(d[-j][i]@unweightspg)>1):
                        j += 1
                    conv -= d[-j][i]
        return sponge@conv
    

    def subRookCtl(self, color, lst, rooks = None): 
        unweightspg = self.sponge(lst)
        weighted = self.getWeighted(lst)
        if(color==True):
            if(rooks is None):
                rooks = lst[3*64:4*64]
            advKing = lst[11*64:12*64]
        else:
            if(rooks is None):
                rooks = lst[9*64:10*64]
            advKing = lst[5*64:6*64]
        sponge = abs(self.sponge(weighted)) + 9*advKing
        conv = np.asarray(64*[0])
        dir = [self.strEl.S, self.strEl.N, self.strEl.E, self.strEl.W]
        for i in range(64):
            for i in range(64):
                if(rooks[i]==1):
                    for d in dir:
                        j = 0
                        while(np.sum(d[-j][i]@unweightspg)>1):
                            j += 1
                        conv += d[-j][i]
                elif(rooks[i]==-1):
                    for d in dir:
                        j = 0
                        while(np.sum(d[-j][i]@unweightspg)>1):
                            j += 1
                        conv -= d[-j][i]        
        return sponge@conv
                    
    def subKnightCtl(self, color, lst, knights = None):
        weighted = self.getWeighted(lst)
        if(color==True):
            if(knights is None):
                knights = lst[64:2*64]
            advKing = lst[11*64:12*64]
        else:
            if(knights is None):
                knights = lst[7*64:8*64]
            advKing = lst[5*64:6*64]
        sponge = abs(self.sponge(weighted)) + 9*advKing
        conv = np.asarray(64*[0])
        for i in range(64):
            if(knights[i]==1):
                conv += self.strEl.Kn[i]
            elif(knights[i]==-1):
                conv -= self.strEl.Kn[i]    
  
        return sponge@conv
    
    def sponge(self, array):
        res = [0]*64
        for i in range(12):
            res = res + array[i*64:(i+1)*64]    
        return res
    
    def setList(self,board):
            # Dictionary to map piece colors to their index for bitboard representation
            self.board = board
            self.__list = np.asarray([0b0]*(768+5))
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if(piece!=None):
                    index = square + (piece.piece_type-1)*64 + COLOR_OFFSET[piece.color]*64*6
                    self.__list[index] = 0b1

            self.__list[-5] = BOOL_BIN[board.turn]
            self.__list[-4] = BOOL_BIN[board.has_kingside_castling_rights(chess.WHITE)]
            self.__list[-3] = BOOL_BIN[board.has_queenside_castling_rights(chess.WHITE)]
            self.__list[-2] = BOOL_BIN[board.has_kingside_castling_rights(chess.BLACK)]
            self.__list[-1] = BOOL_BIN[board.has_queenside_castling_rights(chess.BLACK)]

    def setWeights(self):        
        self.__pieceswght[0:64] = np.asarray([WEIGHTS['PAWN_VALUE']]*64)
        self.__pieceswght[64:128] = np.asarray([WEIGHTS['KNIGHT_VALUE']]*64)
        self.__pieceswght[128:192] = np.asarray([WEIGHTS['BISHOP_VALUE']]*64)
        self.__pieceswght[192:256] = np.asarray([WEIGHTS['ROOK_VALUE']]*64)
        self.__pieceswght[256:320] = np.asarray([WEIGHTS['QUEEN_VALUE']]*64)
        self.__pieceswght[320:384] = np.asarray([0]*64)
        self.__pieceswght[384:448] = np.asarray([-WEIGHTS['PAWN_VALUE']]*64)
        self.__pieceswght[448:512] = np.asarray([-WEIGHTS['KNIGHT_VALUE']]*64)
        self.__pieceswght[512:576] = np.asarray([-WEIGHTS['BISHOP_VALUE']]*64)
        self.__pieceswght[576:640] = np.asarray([-WEIGHTS['ROOK_VALUE']]*64)
        self.__pieceswght[640:704] = np.asarray([-WEIGHTS['QUEEN_VALUE']]*64)
        self.__pieceswght[704:773] = np.asarray([0]*69)

