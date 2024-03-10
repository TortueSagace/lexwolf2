import chess
import numpy as np
from weights import WEIGHTS
from structEl import structEl

class bitBoard:

    def __init__(self,board=None):
        self.COLOR_OFFSET = {chess.WHITE: 0, chess.BLACK: 1}
        self.BOOL_BIN = {True: 1, False: 0}
        self.board = board
        self.__list = np.asarray([None]*(768+5)) # private 772-bits bit array: 64*6*2 + 5 ~ 64 cases for each piece of each color plus side to move & castling rights
        self.__pieceswght = np.asarray([None]*(768+5))
        self.setWeights()
        self.strEl = structEl()
        if(board==None): # no board provided, initial board is supposed
            self.board = chess.Board()

        self.setList(self.board)
        
        self.wghtInit = np.sum(abs(self.getWeighted())) - 16

    def gamePhase(self):
        weighted = abs(self.getWeighted())
        weighted[0:64] = np.asarray(64*[0])
        weighted[6*64:7*64] = np.asarray(64*[0])
        weight = np.sum(weighted)
        return 1 - weight/self.wghtInit

    def getList(self):
        return self.__list
    
    def getWeighted(self):
        return self.__list*self.__pieceswght

    def getEval(self):
        phase = self.gamePhase() # goes from 0% to 100% depending on the weight of non-pawn material remaining
        a1 = 5/6 # material
        a2 = (1-a1)/7 # b ctl
        a3 = (1-a1)/7 # q ctl
        a4 = (1-a1)/7 # r ctl
        a5 = (1-a1)/7 # k ctl
        a6 = (1-a1)/7 # p ctl
        a7 = (1-a1)/14 # p pen
        a8 = (1 -a1)/7 # king exposed
        a9 = (1-a1)/14 # king protected

        if(phase<=0.125):
            res = a1*np.sum(self.getWeighted()) + a5*4.5*self.knightCtl() + a6*self.pawnCtl() - a7*self.pawnPenalty() + a9*self.kingProtected()
        elif(phase>0.125 and phase<=0.25):
            res = a1*np.sum(self.getWeighted()) + a2*self.bishopCtl() + a3*(0.5+0.5*phase)*self.queenCtl() + a5*self.knightCtl() + + phase*a4*self.rookCtl() + a6*self.pawnCtl() - a7*self.pawnPenalty() + a9*self.kingProtected()
        else:
            res = a1*np.sum(self.getWeighted()) + a2*self.bishopCtl() + a3*(0.5+0.5*phase)*self.queenCtl() + phase*a4*self.rookCtl() + a5*self.knightCtl() + a6*self.pawnCtl() - a7*self.pawnPenalty() + a9*self.kingProtected()
            
        return res
    
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
            convw[i]=self.Kg[i]@spongew
            convb[i]=self.Kg[i]@spongeb
        return wKing@convw + bKing@convb    

    def pawnCtl(self):
        weighted = self.getWeighted()
        wKing = self.__list[5*64:6*64]
        bKing = self.__list[11*64:12*64]
        spongew = abs(self.sponge(weighted)) + 9*bKing
        spongeb = abs(self.sponge(weighted)) + 9*wKing
        convw = np.asarray(64*[0])
        convb = np.asarray(64*[0])
        for i in range(64):
            convw[i] = self.strEl.wP[i]@spongew
            convb[i] = self.strEl.bP[i]@spongeb
  
        wPawns = self.__list[0:64]
        bPawns = -self.__list[6*64:7*64]
        
        return wPawns@convw + bPawns@convb

    def pawnPenalty(self):  
        spongew = self.__list[0:64]
        spongeb = self.__list[6*64:7*64]
        convw = np.asarray(64*[0])
        convb = np.asarray(64*[0])
        for i in range(64):
            convw[i] = self.strEl.Pp[i]@spongew
            convb[i] = self.strEl.Pp[i]@spongeb
  
        wPawns = spongew
        bPawns = -spongeb
        
        return wPawns@convw + bPawns@convb  

    def bishopCtl(self):
        weighted = self.getWeighted()
        wKing = self.__list[5*64:6*64]
        bKing = self.__list[11*64:12*64]
        spongew = abs(self.sponge(weighted)) + 9*bKing
        spongeb = abs(self.sponge(weighted)) + 9*wKing
        convw = np.asarray(64*[0])
        convb = np.asarray(64*[0])
        for i in range(64):
            convw[i] = self.strEl.Bp[i]@spongew
            convb[i] = self.strEl.Bp[i]@spongeb
  
        wbishops = self.__list[2*64:3*64]
        bbishops = -self.__list[8*64:9*64]
        
        return wbishops@convw + bbishops@convb
                    
    def knightCtl(self):
        weighted = self.getWeighted()
        wKing = self.__list[5*64:6*64]
        bKing = self.__list[11*64:12*64]
        spongew = abs(self.sponge(weighted)) + 9*bKing
        spongeb = abs(self.sponge(weighted)) + 9*wKing
        convw = np.asarray(64*[0])
        convb = np.asarray(64*[0])
        for i in range(64):
            convw[i] = self.strEl.Kn[i]@spongew
            convb[i] = self.strEl.Kn[i]@spongeb
  
        wknights = self.__list[64:2*64]
        bknights = -self.__list[7*64:8*64]
        
        return wknights@convw + bknights@convb
    
    def queenCtl(self): 
        weighted = self.getWeighted()
        wKing = self.__list[5*64:6*64]
        bKing = self.__list[11*64:12*64]
        spongew = abs(self.sponge(weighted)) + 9*bKing
        spongeb = abs(self.sponge(weighted)) + 9*wKing
        convw = np.asarray(64*[0])
        convb = np.asarray(64*[0])
        for i in range(64):
            convw[i] = self.strEl.Qn[i]@spongew
            convb[i] = self.strEl.Qn[i]@spongeb
  
        wqueen = self.__list[4*64:5*64]
        bqueen = -self.__list[10*64:11*64]
        
        return wqueen@convw+bqueen@convb
    

    def rookCtl(self): 
        weighted = self.getWeighted()
        wKing = self.__list[5*64:6*64]
        bKing = self.__list[11*64:12*64]
        spongew = abs(self.sponge(weighted)) + 9*bKing
        spongeb = abs(self.sponge(weighted)) + 9*wKing
        convw = np.asarray(64*[0])
        convb = np.asarray(64*[0])
        for i in range(64):
            convw[i] = self.strEl.Rk[i]@spongew
            convb[i] = self.strEl.Rk[i]@spongeb
  
        wrooks = self.__list[3*64:4*64]
        brooks = -self.__list[9*64:10*64]
        
        return wrooks@convw+brooks@convb 
    
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
                    index = square + (piece.piece_type-1)*64 + self.COLOR_OFFSET[piece.color]*64*6
                    self.__list[index] = 0b1

            self.__list[-5] = self.BOOL_BIN[board.turn]
            self.__list[-4] = self.BOOL_BIN[board.has_kingside_castling_rights(chess.WHITE)]
            self.__list[-3] = self.BOOL_BIN[board.has_queenside_castling_rights(chess.WHITE)]
            self.__list[-2] = self.BOOL_BIN[board.has_kingside_castling_rights(chess.BLACK)]
            self.__list[-1] = self.BOOL_BIN[board.has_queenside_castling_rights(chess.BLACK)]

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

