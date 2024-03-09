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

    def getList(self):
        return self.__list
    
    def getWeighted(self):
        return self.__list*self.__pieceswght

    def getEval(self):
        a1 = 2/3 # material
        a2 = (1-a1)/3 # b ctl
        a3 = (1-a1)/3 # q ctl
        a4 = (1-a1)/3 # r ctl
        return a1*np.sum(self.getWeighted()) + a2*self.bishopCtl() + a3*self.queenCtl() + a4*self.rookCtl()
    
    def bishopCtl(self):
        res = 0
        al = 1/2 # allied control
        en = 1/2 # ennemy control
        weighted = self.getWeighted()
        if(self.board.turn == chess.WHITE):
            bishops = self.__list[2*64:3*64]
            weighted[2*64:3*64] = np.asarray([0]*64) 
            sponge = self.sponge(weighted)
            allied = (sponge + abs(sponge))/2
            ennemy = (-sponge + abs(sponge))/2
            conva = np.asarray(64*[0])
            conve = np.asarray(64*[0])
            for i in range(64):
                conva[i] = self.strEl.Bp[i]@allied
                conve[i] = self.strEl.Bp[i]@ennemy
            res = (al*conva+en*conve)@bishops

        else:    
            bishops = self.__list[8*64:9*64]
            weighted[8*64:9*64] = np.asarray([0]*64)             
            sponge = self.sponge(weighted)
            allied = (-sponge + abs(sponge))/2
            ennemy = (sponge + abs(sponge))/2
            conva = np.asarray(64*[0])
            conve = np.asarray(64*[0])
            for i in range(64):
                conva[i] = self.strEl.Bp[i]@allied
                conve[i] = self.strEl.Bp[i]@ennemy
            res = (al*conva+en*conve)@bishops

        return res
    
    def queenCtl(self): 
        res = 0
        al = 1/2 # allied control
        en = 1/2 # ennemy control
        weighted = self.getWeighted()
        if(self.board.turn == chess.WHITE):
            queen = self.__list[4*64:5*64]
            weighted[4*64:5*64] = np.asarray([0]*64) 
            sponge = self.sponge(weighted)
            allied = (sponge + abs(sponge))/2
            ennemy = (-sponge + abs(sponge))/2
            conva = np.asarray(64*[0])
            conve = np.asarray(64*[0])
            for i in range(64):
                conva[i] = self.strEl.Qn[i]@allied
                conve[i] = self.strEl.Qn[i]@ennemy
            res = (al*conva+en*conve)@queen

        else:    
            queen = self.__list[10*64:11*64]
            weighted[10*64:11*64] = np.asarray([0]*64)         
            sponge = self.sponge(weighted)
            allied = (-sponge + abs(sponge))/2
            ennemy = (sponge + abs(sponge))/2
            conva = np.asarray(64*[0])
            conve = np.asarray(64*[0])
            for i in range(64):
                conva[i] = self.strEl.Qn[i]@allied
                conve[i] = self.strEl.Qn[i]@ennemy
            res = (al*conva+en*conve)@queen
            
        return res 

    def rookCtl(self):      
        res = 0
        al = 1/2 # allied control
        en = 1/2 # ennemy control
        weighted = self.getWeighted()
        if(self.board.turn == chess.WHITE):
            rooks = self.__list[3*64:4*64]
            weighted[3*64:4*64] = np.asarray([0]*64) 
            sponge = self.sponge(weighted)
            allied = (sponge + abs(sponge))/2
            ennemy = (-sponge + abs(sponge))/2
            conva = np.asarray(64*[0])
            conve = np.asarray(64*[0])
            for i in range(64):
                conva[i] = self.strEl.Rk[i]@allied
                conve[i] = self.strEl.Rk[i]@ennemy
            res = (al*conva+en*conve)@rooks

        else:    
            rooks = self.__list[9*64:10*64]
            weighted[9*64:10*64] = np.asarray([0]*64)         
            sponge = self.sponge(weighted)
            allied = (-sponge + abs(sponge))/2
            ennemy = (sponge + abs(sponge))/2
            conva = np.asarray(64*[0])
            conve = np.asarray(64*[0])
            for i in range(64):
                conva[i] = self.strEl.Rk[i]@allied
                conve[i] = self.strEl.Rk[i]@ennemy
            res = (al*conva+en*conve)@rooks
            
        return res     
    
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

