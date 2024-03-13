import chess
import numpy as np
from weights import WEIGHTS
from structEl import structEl
from math import floor

COLOR_OFFSET = {chess.WHITE: 0, chess.BLACK: 1}
BOOL_BIN = {True: 1, False: 0}

class bitBoard:

    def __init__(self,board=None):
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
        self.__value = None

    def value(self):
        return self.__value
    
    def getList(self):
        return self.__list
    
    def getWeighted(self):
        return self.__list*self.__pieceswght

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
        res = a1*np.sum(self.getWeighted()) + a2*self.bishopCtl() + a3*(0.5+0.5*phase)*self.queenCtl() + phase*a4*self.rookCtl() + (1+0.5*(1-phase))*a5*self.knightCtl() + a6*self.pawnCtl() - a7*self.pawnPenalty() + a9*self.kingProtected()
        self.__value = res
        return res
    
    def getDeltaEval(self, prevBboard):
        res = 0
        listDiff = self.__list - prevBboard.getList()
        listDiff = listDiff[0:768]
        white = True
        index = 0
        for i in range(len(listDiff)):
            if listDiff[i]!=0:
                index = i
        if(i/64>=6):
            white = False
        self.__value = res + prevBboard.value()
        return self.__value
    
    def kingProtected(self):
        whites = self.__list
        whites[6*64:12*64] = np.asarray([0]*6*64)
        spongew = self.sponge(whites)
        blacks = self.__list
        blacks[0:6*64] = np.asarray([0]*6*64)
        spongeb = self.sponge(blacks)
        wKing = self.__list[5*64:6*64]
        bKing = -self.__list[11*64:12*64]
        convw = [0]*64
        convb = [0]*64
        for i in range(64):
            if(wKing[i]==1):
                convw += self.strEl.Kg[i]
            if(bKing[i]==-1):    
                convb -= self.strEl.Kg[i]

        return spongew@convw + spongeb@convb    

    def pawnCtl(self):
        weighted = self.getWeighted()
        wKing = self.__list[5*64:6*64]
        bKing = self.__list[11*64:12*64]
        spongew = abs(self.sponge(weighted)) + 9*bKing
        spongeb = abs(self.sponge(weighted)) + 9*wKing
        convw = np.asarray(64*[0])
        convb = np.asarray(64*[0])
        wPawns = self.__list[0:64]
        bPawns = -self.__list[6*64:7*64]
        
        for i in range(64):
            if(wPawns[i]==1):
                convw += self.strEl.wP[i]
            if(bPawns[i]==-1):    
                convb -= self.strEl.bP[i]

        return spongew@convw + spongeb@convb

    def pawnPenalty(self):  
        spongew = self.__list[0:64]
        spongeb = self.__list[6*64:7*64]
        wPawns = spongew
        bPawns = -spongeb
        convw = np.asarray(64*[0])
        convb = np.asarray(64*[0])
        for i in range(64):
            if(wPawns[i]==1):
                convw += self.strEl.Pp[i]
            if(bPawns[i]==-1):    
                convb -= self.strEl.Pp[i]
        
        return spongew@convw + spongeb@convb  

    def bishopCtl(self):
        weighted = self.getWeighted()
        wKing = self.__list[5*64:6*64]
        bKing = self.__list[11*64:12*64]
        spongew = abs(self.sponge(weighted)) + 9*bKing
        spongeb = abs(self.sponge(weighted)) + 9*wKing
        wbishops = self.__list[2*64:3*64]
        bbishops = -self.__list[8*64:9*64]
        convw = np.asarray(64*[0])
        convb = np.asarray(64*[0])
        for i in range(64):
            if(wbishops[i]==1):
                convw += self.strEl.Bp[i]
            if(bbishops[i]==-1):    
                convb -= self.strEl.Bp[i]
        
        return spongew@convw + spongeb@convb
                    
    def knightCtl(self):
        weighted = self.getWeighted()
        wKing = self.__list[5*64:6*64]
        bKing = self.__list[11*64:12*64]
        spongew = abs(self.sponge(weighted)) + 9*bKing
        spongeb = abs(self.sponge(weighted)) + 9*wKing
        wknights = self.__list[64:2*64]
        bknights = -self.__list[7*64:8*64]
        convw = np.asarray(64*[0])
        convb = np.asarray(64*[0])
        for i in range(64):
            if(wknights[i]==1):
                convw += self.strEl.Kn[i]
            if(bknights[i]==-1):    
                convb -= self.strEl.Kn[i]
  
        return spongew@convw + spongeb@convb
    
    def queenCtl(self): 
        weighted = self.getWeighted()
        wKing = self.__list[5*64:6*64]
        bKing = self.__list[11*64:12*64]
        spongew = abs(self.sponge(weighted)) + 9*bKing
        spongeb = abs(self.sponge(weighted)) + 9*wKing
        convw = np.asarray(64*[0])
        convb = np.asarray(64*[0])
        wqueen = self.__list[4*64:5*64]
        bqueen = -self.__list[10*64:11*64]
        for i in range(64):
            if(wqueen[i]==1):
                convw += self.strEl.Qn[i]
            if(bqueen[i]==-1):    
                convb -= self.strEl.Qn[i]
        
        return spongew@convw+spongeb@convb
    

    def rookCtl(self): 
        weighted = self.getWeighted()
        wKing = self.__list[5*64:6*64]
        bKing = self.__list[11*64:12*64]
        spongew = abs(self.sponge(weighted)) + 9*bKing
        spongeb = abs(self.sponge(weighted)) + 9*wKing
        wrooks = self.__list[3*64:4*64]
        brooks = -self.__list[9*64:10*64]
        convw = np.asarray(64*[0])
        convb = np.asarray(64*[0])
        for i in range(64):
            if(wrooks[i]==1):
                convw += self.strEl.Rk[i]
            if(brooks[i]==-1):    
                convb -= self.strEl.Rk[i]
    
        return spongew@convw+spongew@convb 
    
    def sponge(self, array=None):
        res = [0]*64
        if(not array.any()):
            array = self.__list
        for i in range(12):
            res+=array[i*64:(i+1)*64]    
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

