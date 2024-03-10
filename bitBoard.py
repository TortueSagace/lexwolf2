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
        phase = self.gamePhase # goes from 0% to 100% depending on the weight of non-pawn material remaining
        a1 = 5/6 # material
        a2 = (1-a1)/5 # b ctl
        a3 = (1-a1)/5 # q ctl
        a4 = (1-a1)/5 # r ctl
        a5 = 2*(1-a1)/5 # k ctl
        return a1*np.sum(self.getWeighted()) + a2*self.bishopCtl() + a3*(0.5+0.5*phase)*self.queenCtl() + phase*a4*self.rookCtl() + a5*self.knightCtl()
    
    def bishopCtl(self):
        weighted = self.getWeighted()
        sponge = abs(self.sponge(weighted))
        conv = np.asarray(64*[0])
        for i in range(64):
            conv[i] = self.strEl.Bp[i]@sponge
  
        wbishops = self.__list[2*64:3*64]
        bbishops = -self.__list[8*64:9*64]
        
        return (wbishops+bbishops)@conv
    
    def knightCtl(self):
        weighted = self.getWeighted()
        sponge = abs(self.sponge(weighted))
        conv = np.asarray(64*[0])
        for i in range(64):
            conv[i] = self.strEl.Kn[i]@sponge
  
        wknights = self.__list[64:2*64]
        bknights = -self.__list[7*64:8*64]
        
        return (wknights+bknights)@conv
    
    def queenCtl(self): 
        weighted = self.getWeighted()
        sponge = abs(self.sponge(weighted))
        conv = np.asarray(64*[0])
        for i in range(64):
            conv[i] = self.strEl.Qn[i]@sponge
  
        wqueen = self.__list[4*64:5*64]
        bqueen = -self.__list[10*64:11*64]
        
        return (wqueen+bqueen)@conv
    

    def rookCtl(self):   
        weighted = self.getWeighted()
        sponge = abs(self.sponge(weighted))
        conv = np.asarray(64*[0])
        for i in range(64):
            conv[i] = self.strEl.Rk[i]@sponge
  
        wrooks = self.__list[3*64:4*64]
        brooks = -self.__list[9*64:10*64]
        
        return (wrooks+brooks)@conv   
    
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

