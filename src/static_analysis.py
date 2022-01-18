import math
import numpy as np
import scipy
import scipy.linalg
from scipy.linalg import solve_triangular
import csv
import time
# Input Data
import os

# ############################# Get Project Data #############################
dir_path = os.path.dirname(os.path.realpath(__file__))
project_name = "mahini_softening"
project_input_directory = os.path.join(dir_path, "Input_Data", project_name)

# Structure Geometry Data:
# n: No. of Nodes
# L: No. of Element
n_L = np.loadtxt(os.path.join(project_input_directory, "structure_geometry", "n_L.txt")).astype(int)
n = n_L[0]
L = n_L[1]

# LOEN: Label of Element's Node
# Element Label, Beginning Node Label, End Node Label" and Type of End of Elements
# End of an Element Can be Fixed or Hinge. So We Have 4 type of Element:
# 0- Fixed-Fixed
# 1- Hinged-Fixed
# 2- Fixed-Hinged
# 3- Hinged-Hinged
LOEN = np.matrix((np.loadtxt(os.path.join(project_input_directory, "structure_geometry", "LOEN.txt"), usecols=range(4), delimiter=",")).astype(int))

# COE: Coordinate of Elements
# Element Label, x1, y1, x2, y2"
COE = np.matrix(np.loadtxt(os.path.join(project_input_directory, "structure_geometry", "COE.txt"), usecols=range(5), delimiter=","))
# ---------------------------------------------------------------------------

# Material Properties Data:
# MOE: Modulus of Elasticity
MOE = np.loadtxt(os.path.join(project_input_directory, "material", "MOE.txt"))

# AI: Element's Info: Area and Moment of Inertia of Elements
# Area of Element, Moment of Inertia of Element
AI = np.matrix(np.loadtxt(os.path.join(project_input_directory, "material", "AI.txt"), usecols=range(2), delimiter=","))

# section_capacity: Flextural Capacity of The Element
section_capacity = np.matrix(np.loadtxt(os.path.join(project_input_directory, "material", "section_capacity.txt"), delimiter=","))
# ---------------------------------------------------------------------------

# Boundary Condition Data:
# nR: No. of Restraints
nR = np.loadtxt(os.path.join(project_input_directory, "boundary", "nR.txt")).astype(int)

# JTR: Joint and Type of Restraint
# "Enter: Joint Label, Type of Restraint ( 0: X Direction Reaction 1: Y Direction Reaction 2: Moment Reaction )"
JTR = np.matrix((np.loadtxt(os.path.join(project_input_directory, "boundary", "JTR.txt"), usecols=range(2), delimiter=",")).astype(int))
# ---------------------------------------------------------------------------

# Loading Data:
# nCJL: Number of Concentrated Joint Load:
nCJL = np.loadtxt(os.path.join(project_input_directory, "loading", "nCJL.txt")).astype(int)

# CJL: Concentrated Joint Load:
# Label of Joint, Type of Load (0:X Direction Load 1:Y Direction Load 2:Momemnt Load), Magnitude of Loaad
CJL = np.matrix(np.loadtxt(os.path.join(project_input_directory, "loading", "CJL.txt"), usecols=range(3), ndmin=2, delimiter=","))

# lambdax: load multiplier:
lambdax = np.loadtxt(os.path.join(project_input_directory, "loading", "lambdax.txt"))
# lambdax = 287000.0


# delta_limit: Maximum Displacement limit for  Certain Degree of Freedom
# 0: Node Number
# 1: Degree of Freedom (0: X, 1: Y, 2: Z)
# 2: Magnitude
delta_limit = np.matrix(np.loadtxt(os.path.join(project_input_directory, "loading", "delta_limit.txt"), usecols=range(3), ndmin=2, delimiter=","))
print("delta_limit", delta_limit)

# nCEL: Number of Concentrated Member Load:
nCEL = np.loadtxt(os.path.join(project_input_directory, "loading", "nCEL.txt")).astype(int)

# Concentrated Member Load:
CEL = np.matrix(np.loadtxt(os.path.join(project_input_directory, "loading", "CEL.txt"), usecols=range(4), delimiter=","))

# Example: CEL=np.matrix([1,2,4.0,0])

# nDL: Number of Distributed Load:
nDL = np.loadtxt(os.path.join(project_input_directory, "loading", "nDL.txt")).astype(int)

# Distributed Load:
DL = np.matrix(np.loadtxt(os.path.join(project_input_directory, "loading", "DL.txt"), usecols=range(5), delimiter=","))
# Example: DL = np.matrix([0, 2, 0, 10, 0])

# nM: Number of Mass Spans
nM = np.loadtxt(os.path.join(project_input_directory, "loading", "nM.txt")).astype(int)

# EoM: Element that has Mass on it
EoM = np.matrix(np.loadtxt(os.path.join(project_input_directory, "loading", "EoM.txt"), delimiter=","))

# MoM: Magnitude of Mass
MoM = np.matrix(np.loadtxt(os.path.join(project_input_directory, "loading", "MoM.txt")))
# ---------------------------------------------------------------------------
# ####################### End of Getting Project Data #######################

# Defining Loadings:
Ft = np.zeros((3*n, 1))

# Ff: Fixed End Forces Due to Member Forces
Ff = np.zeros((L, 6))

# nom: Counter for No. of Mass

# Definition of Yield Surface
# Computing Number of Plasticity Susceptible Sections
nPS = 2*L  # برای راحت تر شدن مسئله همه دو سر اعضا را به عنوان مقطع مستعد خمیری شدن در نظر میگیریم
# for i in range(L):
#     if LOEN[i,3] == 1:
#         nPS -= 1
#     elif LOEN[i,3] == 2:
#         nPS -= 1
#     elif LOEN[i,3] == 3:
#         nPS -= 2
# print(nPS)
#  Definition of Yield Surface
# phiP0=matmul(transpose(Yield),P0) !Calculation OK.
# phipvphi=matmul(transpose(Yield),matmul(UDF,Yield)) !Calculation OK.


sectionCapacity = np.zeros((L, 2))
for eln in range(L):
    sectionCapacity[eln, 0] = -section_capacity[0, eln]
    sectionCapacity[eln, 1] = section_capacity[0, eln]

phi = np.zeros((2*L, 4*L))
for i in range(L):
    phi[2*i,4*i]=1/sectionCapacity[i,0]
    phi[2*i,4*i+1]=1/sectionCapacity[i,1]
    phi[2*i+1,4*i+2]=1/sectionCapacity[i,0]
    phi[2*i+1,4*i+3]=1/sectionCapacity[i,1]

if project_name == "moharrami_new_method":
    phi[2, 4] = -1/42500
    phi[2, 5] = 1/42500
    phi[5, 10] = -1/42500
    phi[5, 11] = 1/42500


if project_name == "mahini_softening":
    phi[1, 2] = -1/420.7e3
    phi[1, 3] = 1/420.7e3
    phi[3, 6] = -1/420.7e3
    phi[3, 7] = 1/420.7e3
    phi[5, 10] = -1/420.7e3
    phi[5, 11] = 1/420.7e3
    phi[8, 16] = -1/236.8e3
    phi[8, 17] = 1/236.8e3
    phi[10, 20] = -1/236.8e3
    phi[10, 21] = 1/236.8e3


# Computing Length of elements
def LEN_Maker(x1, y1, x2, y2):
    LEN = np.sqrt((x2-x1)**2.0 + (y2-y1)**2.0)
    return LEN
# allLen stores length of all of elements
allLen=[]
for i in range(L):
    allLen.append(LEN_Maker(COE[i,1],COE[i,2],COE[i,3],COE[i,4]))

# ai=np.zeros((1,3))
# # ai:Area, I of section, Type of element by end type
# ai[0,0:2]=AI[eln,:]
# ai[0,2]=LOEN[eln,3]
# print(LOEN[0,3])
# print(LOEN[1,3])
 # Computing Transformation Matrix
def T_Maker(x1,y1,x2,y2,LEN):
    T=np.matrix([[(x2-x1)/LEN, (y2-y1)/LEN, 0.0, 0.0, 0.0, 0.0],
    [-(y2-y1)/LEN, (x2-x1)/LEN, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, (x2-x1)/LEN, (y2-y1)/LEN, 0.0],
    [0.0, 0.0, 0.0, -(y2-y1)/LEN, (x2-x1)/LEN, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    return T
# allT stores Transform matrix of all of elements
allT=[]
for i in range(L):
    allT.append(T_Maker(COE[i,1],COE[i,2],COE[i,3],COE[i,4],allLen[i]))

# Computation of Stiffness Matrix and Unit Equilibrated Forces According to Type of its ends
def k_Maker(MOE, LEN, ai, supportType):
    if (supportType == 0):
        K = np.matrix([
            [MOE*ai[0,0]/LEN, 0.0, 0.0, -MOE*ai[0,0]/LEN, 0.0, 0.0],
            [0.0, 12.0*MOE*ai[0,1]/(LEN**3.0), 6.0*MOE*ai[0,1]/(LEN**2.0), 0.0, -12.0*MOE*ai[0,1]/(LEN**3.0), 6.0*MOE*ai[0,1]/(LEN**2.0)],
            [0.0, 6.0*MOE*ai[0,1]/(LEN**2.0), 4.0*MOE*ai[0,1]/(LEN), 0.0, -6.0*MOE*ai[0,1]/(LEN**2.0), 2.0*MOE*ai[0,1]/(LEN)],
            [-MOE*ai[0,0]/LEN, 0.0, 0.0, MOE*ai[0,0]/LEN, 0.0, 0.0],
            [0.0, -12.0*MOE*ai[0,1]/(LEN**3.0), -6.0*MOE*ai[0,1]/(LEN**2.0), 0.0, 12.0*MOE*ai[0,1]/(LEN**3.0), -6.0*MOE*ai[0,1]/(LEN**2.0)],
            [0.0, 6.0*MOE*ai[0,1]/(LEN**2.0), 2.0*MOE*ai[0,1]/(LEN), 0.0, -6.0*MOE*ai[0,1]/(LEN**2.0), 4.0*MOE*ai[0,1]/(LEN)]])

    elif (supportType == 1):
        K = np.matrix([
            [MOE*ai[0,0]/LEN, 0.0, 0.0, -MOE*ai[0,0]/LEN, 0.0, 0.0],
            [0.0, 3.0*MOE*ai[0,1]/(LEN**3.0), 0.0, 0.0, -3.0*MOE*ai[0,1]/(LEN**3.0), 3.0*MOE*ai[0,1]/(LEN**2.0)],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-MOE*ai[0,0]/LEN, 0.0, 0.0, MOE*ai[0,0]/LEN, 0.0, 0.0],
            [0.0, -3.0*MOE*ai[0,1]/(LEN**3.0), 0.0, 0.0, 3.0*MOE*ai[0,1]/(LEN**3.0), -3.0*MOE*ai[0,1]/(LEN**2.0)],
            [0.0, 3.0*MOE*ai[0,1]/(LEN**2.0), 0.0, 0.0, -3.0*MOE*ai[0,1]/(LEN**2.0), 3.0*MOE*ai[0,1]/(LEN)]])

    elif (supportType == 2):
        K = np.matrix([
            [MOE*ai[0,0]/LEN, 0.0, 0.0, -MOE*ai[0,0]/LEN, 0.0, 0.0],
            [0.0, 3.0*MOE*ai[0,1]/(LEN**3.0), 3.0*MOE*ai[0,1]/(LEN**2.0), 0.0, -3.0*MOE*ai[0,1]/(LEN**3.0), 0.0],
            [0.0, 3.0*MOE*ai[0,1]/(LEN**2.0), 3.0*MOE*ai[0,1]/(LEN), 0.0, -3.0*MOE*ai[0,1]/(LEN**2.0), 0.0],
            [-MOE*ai[0,0]/LEN, 0.0, 0.0, MOE*ai[0,0]/LEN, 0.0, 0.0],
            [0.0, -3.0*MOE*ai[0,1]/(LEN**3.0), -3.0*MOE*ai[0,1]/(LEN**2.0), 0.0, 3.0*MOE*ai[0,1]/(LEN**3.0), 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    elif (supportType == 3):
        K = np.matrix([
            [MOE*ai[0,0]/LEN, 0.0, 0.0, -MOE*ai[0,0]/LEN, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-MOE*ai[0,0]/LEN, 0.0, 0.0, MOE*ai[0,0]/LEN, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    UDL = K[:, [2, 5]]
    return K, UDL
# allK stores stiffness of all of elements
allK=[]
allUDL=[]


for i in range(L):
    k,UDL=k_Maker(MOE,allLen[i],AI[i],int(LOEN[i,3]))
    allK.append(k)
    allUDL.append(UDL)    
    allUDL[i][:,0]=np.dot(np.transpose(allT[i]),allUDL[i][:,0])
    allUDL[i][:,1]=np.dot(np.transpose(allT[i]),allUDL[i][:,1])
np.set_printoptions(linewidth=100)
np.set_printoptions(precision=4)

#نحوه فراخوانی نیروهای خودمتعادل : allUDL[اندیس عضو مورد نظر][:,0 یا 1]

# for i in range(2):
#     Kg=np.dot(np.dot(np.transpose(allT[i]),allK[i]),allT[i])
#     print(Kg)
# Computing Joint Masses
# M=np.zeros((3*n,3*n))
# for nom in range(nM):
#     eln=EoM[0,nom]
#     x1=COE[eln,1]
#     y1=COE[eln,2]
#     x2=COE[eln,3]
#     y2=COE[eln,4]
#     LEN=LEN_Maker(x1,y1,x2,y2)
#     Mnode=MoM[0,nom]*LEN/2.0
#     M[3*LOEN[eln,1]-3,3*LOEN[eln,1]-3]=M[3*LOEN[eln,1]-3,3*LOEN[eln,1]-3]+Mnode
#     M[3*LOEN[eln,1]-2,3*LOEN[eln,1]-2]=M[3*LOEN[eln,1]-2,3*LOEN[eln,1]-2]+Mnode
#     M[3*LOEN[eln,2]-3,3*LOEN[eln,2]-3]=M[3*LOEN[eln,2]-3,3*LOEN[eln,2]-3]+Mnode
#     M[3*LOEN[eln,2]-2,3*LOEN[eln,2]-2]=M[3*LOEN[eln,2]-3,3*LOEN[eln,2]-2]+Mnode
# Applying Boundary Conditions to Stiffness Matrix or Force Vector
def apply_Boundry_Matrix(JTR,M):
    MR=M
    jj=0
    Ft.shape[1]
    if M.shape[1] == 1:
        for BC in range(len(JTR)):
            MR=np.delete(MR,3*JTR[BC,0]+JTR[BC,1]-jj,0) #delete row 1
            jj+=1
    elif M.shape[1] != 1:
        for BC in range(len(JTR)):
            MR=np.delete(MR,3*JTR[BC,0]+JTR[BC,1]-jj,1) #delete column 1
            MR=np.delete(MR,3*JTR[BC,0]+JTR[BC,1]-jj,0) #delete row 1
            jj+=1
    return MR
# # Computation of Structure Stiffness in Global Coordinate
# Kt=np.zeros((3*n,3*n))
# for eln in range (0,L):
# ########### Computation of Element Stiffness in Global Coordinate ###########
#     Kg=np.dot(np.dot(np.transpose(allT[eln]),allK[eln]),allT[eln])
# ########################### Assembling Stiffness ###########################
#     for i in range(0,6):
#         for j in range(0,6):
#             ndn=(j)//3
#             p=3*LOEN[eln,ndn+1]+j%3
#             ndnn=(i)//3
#             q=3*LOEN[eln,ndnn+1]+i%3
#             Kt[p,q]=Kt[p,q]+Kg[j,i]
Kt = np.zeros((3*n, 3*n))
Kt = assemble_2d_frame(L, n, allT, allK, LOEN, Kt)

KR=apply_Boundry_Matrix(JTR,Kt)
np.set_printoptions(linewidth=200)
np.set_printoptions(precision=6)

def CJL_Maker(Ft,JL):
    if (JL[0,1]==0):
        Ft[3*int(JL[0,0])]=Ft[3*int(JL[0,0])]+JL[0,2]
    elif (JL[0,1]==1):
        Ft[3*int(JL[0,0])+1]=Ft[3*int(JL[0,0])+1]+JL[0,2]
    elif (JL[0,1]==2):
        Ft[3*int(JL[0,0])+2]=Ft[3*int(JL[0,0])+2]+JL[0,2]
    return Ft
for jN in range(nCJL):
    CJL_Maker(Ft, CJL[jN])

## Computation of Concentrated Member Load
def CEL_Maker(CEL):
    F=np.zeros((6,1))
    eln=int(CEL[0,0])
    x1=COE[eln,1]
    y1=COE[eln,2]
    x2=COE[eln,3]
    y2=COE[eln,4]
    LEN=LEN_Maker(x1,y1,x2,y2)
#     T=T_Maker(x1,y1,x2,y2,LEN)
    a=CEL[0,2]
    b=LEN-a
    #Axial Force in a distance equal to a from left of element
    if CEL[0,1] == 1:
        F[0,0]=-CEL[0,3]*b/LEN
        F[1,0]=0.0
        F[2,0]=0.0
        F[3,0]=-CEL[0,3]*a/LEN
        F[4,0]=0.0
        F[5,0]=0.0
    #Transverse Force in a distance equal to a from left of element
    elif CEL[0,1] == 2:
        F[0,0]=0.0
        F[1,0]=-CEL[0,3]*(b**2)*(3*a+b)/(LEN**3)
        F[2,0]=-CEL[0,3]*a*(b**2)/(LEN**2)
        F[3,0]=0.0
        F[4,0]=-CEL[0,3]*(a**2)*(3*a+b)/(LEN**3)
        F[5,0]=CEL[0,3]*b*(a**2)/(LEN**2)
    #Concentrated Moment in a distance equal to a from left of element
    elif CEL[0,1] == 3:
        F[0,0]=0.0
        F[1,0]=6.0*CEL[0,3]*a*b/(LEN**3)
        F[2,0]=CEL[0,3]*b*(2*a-b)/(LEN**2)
        F[3,0]=0.0
        F[4,0]=-6.0*CEL[0,3]*a*b/(LEN**3)
        F[5,0]=CEL[0,3]*a*(2*b-a)/(LEN**2)
    return F
#Assembling Concetrated Member Loads
for jN in range(nCEL):
    Q=CEL_Maker(CEL[jN,:])
    x1=COE[int(CEL[jN,0]),1]
    y1=COE[int(CEL[jN,0]),2]
    x2=COE[int(CEL[jN,0]),3]
    y2=COE[int(CEL[jN,0]),4]
    LEN=LEN_Maker(x1,y1,x2,y2)
    T=T_Maker(x1,y1,x2,y2,LEN)
    #Transformation of Local Force to global (Page 272 Kassimali): F=TTQ
    F=np.dot(np.transpose(T),Q)
    Ft[3*int(LOEN[int(CEL[jN,0]),1]),0]=Ft[3*int(LOEN[int(CEL[jN,0]),1]),0]-F[0,0]
    Ft[3*int(LOEN[int(CEL[jN,0]),1])+1,0]=Ft[3*int(LOEN[int(CEL[jN,0]),1])+1,0]-F[1,0]
    Ft[3*int(LOEN[int(CEL[jN,0]),1])+2,0]=Ft[3*int(LOEN[int(CEL[jN,0]),1])+2,0]-F[2,0]
    Ft[3*int(LOEN[int(CEL[jN,0]),2]),0]=Ft[3*int(LOEN[int(CEL[jN,0]),2]),0]-F[3,0]
    Ft[3*int(LOEN[int(CEL[jN,0]),2])+1,0]=Ft[3*int(LOEN[int(CEL[jN,0]),2])+1,0]-F[4,0]
    Ft[3*int(LOEN[int(CEL[jN,0]),2])+2,0]=Ft[3*int(LOEN[int(CEL[jN,0]),2])+2,0]-F[5,0]
    Ff[int(CEL[jN,0]),0]=Ff[int(CEL[jN,0]),0]+Q[0,0]
    Ff[int(CEL[jN,0]),1]=Ff[int(CEL[jN,0]),1]+Q[1,0]
    Ff[int(CEL[jN,0]),2]=Ff[int(CEL[jN,0]),2]+Q[2,0]
    Ff[int(CEL[jN,0]),3]=Ff[int(CEL[jN,0]),3]+Q[3,0]
    Ff[int(CEL[jN,0]),4]=Ff[int(CEL[jN,0]),4]+Q[4,0]
    Ff[int(CEL[jN,0]),5]=Ff[int(CEL[jN,0]),5]+Q[5,0]


## Computation of Distributed Member Load
def DL_Maker(DL):
    F=np.zeros((6,1))
    eln=int(DL[0,0])
    x1=COE[eln,1]
    y1=COE[eln,2]
    x2=COE[eln,3]
    y2=COE[eln,4]
    LEN=LEN_Maker(x1,y1,x2,y2)
    L1=DL[0,2]
    if DL[0,1] == 1:
        L2=LEN-DL[0,3]
        F[0,0]=-DL[0,4]*(LEN-L1-L2)*(LEN-L1+L2)/(2*LEN)
        F[1,0]=0.0
        F[2,0]=0.0
        F[3,0]=-DL[0,4]*(LEN-L1-L2)*(L+L1-L2)/(2*LEN)
        F[4,0]=0.0
        F[5,0]=0.0
    elif DL[0,1] == 2:
        L2=DL[0,3]
        d=L2-L1
        a=0.5*(L1+L2)
        b=0.5*(2*LEN-L1-L2)
        F[0,0]=0.0
        F[1,0]=(DL[0,4]*d/(LEN**3))*((2*a+LEN)*b**2+((a-b)/4)*d**2)
        F[2,0]=(DL[0,4]*d/(LEN**2))*(a*b**2+((a-2*b)*d**2)/12)
        F[3,0]=0.0
        F[4,0]=(DL[0,4]*d/(LEN**3))*((2*b+LEN)*a**2-((a-b)/4)*d**2)
        F[5,0]=(-DL[0,4]*d/(LEN**2))*(a**2*b+((b-2*a)*d**2)/12)
#         print("F",F)
    return F
#Assembling Distributed Member Loads
for jN in range(nDL):
    Q=DL_Maker(DL[jN,:])
    x1=COE[int(DL[jN,0]),1]
    y1=COE[int(DL[jN,0]),2]
    x2=COE[int(DL[jN,0]),3]
    y2=COE[int(DL[jN,0]),4]
    LEN=LEN_Maker(x1,y1,x2,y2)
    #Transformation of Local Force to global (Page 272 Kassimali): F=TTQ
    T=T_Maker(x1,y1,x2,y2,LEN)
    F=np.dot(np.transpose(T),Q)
    Ft[3*int(LOEN[int(DL[jN,0]),1]),0]=Ft[3*int(LOEN[int(DL[jN,0]),1]),0]-F[0,0]
    Ft[3*int(LOEN[int(DL[jN,0]),1])+1,0]=Ft[3*int(LOEN[int(DL[jN,0]),1])+1,0]-F[1,0]
    Ft[3*int(LOEN[int(DL[jN,0]),1])+2,0]=Ft[3*int(LOEN[int(DL[jN,0]),1])+2,0]-F[2,0]
    Ft[3*int(LOEN[int(DL[jN,0]),2]),0]=Ft[3*int(LOEN[int(DL[jN,0]),2]),0]-F[3,0]
    Ft[3*int(LOEN[int(DL[jN,0]),2])+1,0]=Ft[3*int(LOEN[int(DL[jN,0]),2])+1,0]-F[4,0]
    Ft[3*int(LOEN[int(DL[jN,0]),2])+2,0]=Ft[3*int(LOEN[int(DL[jN,0]),2])+2,0]-F[5,0]
    Ff[int(DL[jN,0]),0]=Ff[int(DL[jN,0]),0]+Q[0,0]
    Ff[int(DL[jN,0]),1]=Ff[int(DL[jN,0]),1]+Q[1,0]
    Ff[int(DL[jN,0]),2]=Ff[int(DL[jN,0]),2]+Q[2,0]
    Ff[int(DL[jN,0]),3]=Ff[int(DL[jN,0]),3]+Q[3,0]
    Ff[int(DL[jN,0]),4]=Ff[int(DL[jN,0]),4]+Q[4,0]
    Ff[int(DL[jN,0]),5]=Ff[int(DL[jN,0]),5]+Q[5,0]

FtR=apply_Boundry_Matrix(JTR,Ft)

# Computation of Nodal Displacements
#Cholseky Decomposition Lower Triangular
ca = scipy.linalg.cholesky(KR, lower=True)
#Solving Cholesky by backsubstitution
def cholsl(ca,b):
    n=b.shape[0]
    x=np.zeros((n,1))
    for i in range(n):
        sum=b[i,0]
        for k in range(i-1,-1,-1):
            sum=sum-ca[i,k]*x[k,0]
        x[i,0]=sum/ca[i,i]
    for i in range(n-1,-1,-1):
        sum=x[i,0]
        for k in range(i+1,n):
            sum=sum-ca[k,i]*x[k,0]
        x[i,0]=sum/ca[i,i]
    return x
RDeltaR=cholsl(ca,FtR)

# Computation of Displacement of Structure
def total_nodal_response(n,nR,JTR,RDeltaR):
    j=0
    o=0
    Deltat=np.zeros((3*n,1))
    for i in range(3*n):
        if (j != nR and i == 3*JTR[j,0]+JTR[j,1]):
            j+=1
        else:
            Deltat[i,0]=RDeltaR[o,0]
            o+=1
    return Deltat
Deltat=total_nodal_response(n,nR,JTR,RDeltaR)        

# Computation of Member Forces
def memberForce(LOEN,k,T,Deltat,Ff):
    V=np.zeros((6,1))
    V[0,0]=Deltat[3*LOEN[0,1],0]
    V[1,0]=Deltat[3*LOEN[0,1]+1,0]
    V[2,0]=Deltat[3*LOEN[0,1]+2,0]
    V[3,0]=Deltat[3*LOEN[0,2],0]
    V[4,0]=Deltat[3*LOEN[0,2]+1,0]
    V[5,0]=Deltat[3*LOEN[0,2]+2,0]
    U=np.dot(T,V)
    Fs=np.transpose(np.dot(k,U)+Ff)
    return Fs

Fs=np.zeros((L,6))
P0=np.zeros((2*L,1))
for eln in range(L):
    Ff1=np.reshape(Ff[eln,:], (6, 1))
    Fs[eln,:]=(memberForce(LOEN[eln,:],allK[eln],allT[eln],Deltat,Ff1))
    P0[2*eln]=Fs[eln,2]
    P0[2*eln+1]=Fs[eln,5]
#     Pv[2*eln,elnn]=Fs1[eln,2]
#     Pv[2*eln+1,elnn]=Fs1[eln,5]
#If we want to use only the Plastic Hinges for Unit deformation forces
# """
# # Computation of number of elements that has the capability of being Plastic Hinges
# nS=2*L
# for i in range(L):
#     if LOEN[i,3] == 1 or LOEN[i,3] == 2:
#         nS=nS-1
#     elif LOEN[i,3] == 3:
#         nS=nS-2
# """
# #
# """ 
# LOEN=np.matrix([[1,0,1,2],
# [2,1,2,2],
# [3,2,3,2],
# [4,3,4,1]])

# xs=np.matrix([[3.2],[2],[6.2],[8.1]])
# print(xs)
# x=np.zeros((2*L,1))
# ns=0
# # مشخص کردن اینکه ضرایب پلاستیک محاسبه شده متناظر با کدام مقاطع هستند
# for i in range(L):
#     if LOEN[i,3] == 1:
#         x[2*i,0]=0
#         x[2*i+1,0]=xs[2*i-ns,0]
#         ns += 1
#     elif LOEN[i,3] == 2:
#         x[2*i,0]=xs[2*i-ns,0]
#         x[2*i+1,0]=0
#         ns += 1
#     elif LOEN[i,3] == 3:
#         x[2*i,0]=0
#         ns += 1
#         x[2*i+1,0]=0
#         ns += 1
#     else:
#         x[2*i,0]=xs[2*i-ns,0]
#         x[2*i+1,0]=xs[2*i+1-ns,0]
# print(x)"""
# Computation of Pv
#At first we must build total force vector for each equilibrated force:
ns=2*L
allUnitDeltat=[]
unitMemberForce=[]
Pv=np.zeros((2*L,2*L))
for elnn in range(ns):
    Ft1=np.zeros((3*n,1))
    FfPv=np.zeros((L,6))
    i=elnn//2
    Ft1[3*LOEN[i,1]]=allUDL[i][0,elnn%2]
    Ft1[3*LOEN[i,1]+1]=allUDL[i][1,elnn%2]
    Ft1[3*LOEN[i,1]+2]=allUDL[i][2,elnn%2]
    Ft1[3*LOEN[i,2]]=allUDL[i][3,elnn%2]
    Ft1[3*LOEN[i,2]+1]=allUDL[i][4,elnn%2]
    Ft1[3*LOEN[i,2]+2]=allUDL[i][5,elnn%2]
    FtR1=apply_Boundry_Matrix(JTR,Ft1)
    
    RDeltaR1=cholsl(ca,FtR1)
    
    Deltat1=total_nodal_response(n,nR,JTR,RDeltaR1)

    allUnitDeltat.append(Deltat1) #How to recall allUnitDeltat[elnn][:]
    Fs1=np.zeros((L,6))
    FfPv[i,0:6]=np.reshape(-allUDL[i][:,elnn%2], (1, 6))
    for eln in range(L):
        Fs1[eln,:]=(memberForce(LOEN[eln,:],allK[eln],allT[eln],Deltat1,np.reshape(FfPv[eln,0:6], (6,1))))
        Pv[2*eln,elnn]=Fs1[eln,2]
        Pv[2*eln+1,elnn]=Fs1[eln,5]
    unitMemberForce.append(Fs1) #How to recall each element of unit force Member force: unitMemberForce[elnn]

#     call Elastic_Deform(L,n,LOEN,MOE,AI,COE,Deltat1,Deltas1)
#     deltast(1:L,1:6,elnn)=Deltas1(1:L,1:6)

# Mathematical Programming 
## Creating the Problem
# !############## Determining the Raw Data for Optimization #################
nV=4*L+1 #Number of Variables
nC=4*L+1 #Number of Constraints
A=np.zeros((4*L+1,4*L+1))
phiP0=np.dot(np.transpose(phi),P0)
phiPvPhi=np.dot(np.transpose(phi),np.dot(Pv,phi))


np.savetxt("phiPvPhi.csv", phiPvPhi, delimiter=",")
np.savetxt("phiP0.csv", phiP0, delimiter=",")
A[0:4*L,0:4*L]=phiPvPhi[0:4*L,0:4*L]
A[0:4*L,4*L]=phiP0[0:4*L,0]
A[4*L,4*L]=1.0
b=np.ones((nC,1))
b[nC-1,0]=lambdax
minmax=2
equ=np.ones((nC,1))
C=np.zeros(2*nV)
C[0:4*L]=1.0
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", b)

## Complementarity Programing Revised Simplex
def complementarity_programming(nC,nV,A,b,minmax,equ,C):
    print("***********************\n", A)
# CA==1: hameye C haye manfi daraye a namosbat hastand
# CA==2: hameye C haye manfi daraye a namosbat nistand
# nC: No. of Constraint Equations
# nV: No. of Variables
# minmax: A parameter which declares that optimization is minimization of maximization
# 1: Minimization   2: Maximization
# equ: is a parameter to show the constraint is equality or unequality
# 1: Less Than or Equal  2: Larger Than or Eqaul  3: Equal
# A: Matrix of Coefficient of Variables
# b: Vector of Constants
# C: Vector of Cost Factors
# f: Value of optimization
# dn: Parameter to check that d is nonnegative or not. 1: Negative, 2: Nonnegative
# ba: bi/ais
# minba: minimume of bi/ais
# bV: Basic Variable no.
# !jj: A Parameter that in each iteration if xy==0 its value will increase by 1. At the first jj=0.
# !xy: A parameter to distinguish wheather if x is in basis, y=xs or not? xy==0: x in basis and y=xs, vice versa. xy==1: x in basis and y/=Xs, vice versa. 
# ------------------------------------Construction of Aj
# ---------------------------Apply Inequality Conditions
    for i in range(nC):
        if equ[i]==2:
            b[i]=-b[i]
            A[i,0:nV]=-A[i,0:nV]
# ===================End of Apply Inequality Conditions
# -------------Creating Minimization Objective Function
    if minmax==2: #2: Maximization
        C[0:nV]=-C[0:nV]
# ======End of Creating Minimization Objective Function
    
# Chenanche ma bekhahim mohasebat ra bar asase A voroodi va moteghayyerhaye x va y anjam dahim, be dalile vojoode shartheye
# Complementarity hichyek az moteghayerhaye x varede basis namishavand va amalan hichgoone mohasebati anjam namishavad va amaliate
# behinesazi be etmam miresad va ma be hich javabi dast peyda namikonim. Baraye inke in moshkel ra raf' konim khodeman Landax ra be
# onvane voroodi dar nazar migirim va varede basis mikonim. ba vared shodan landa be basis, dar vaghe' engar saze ra bargozari kardeim.
# agar landa varede moteghayerhaye voroodi nashode bashed, mesle in ast ke saze aslan bargozari nashode ast.

# ----------------------------Creation of Aj Matrix
# the first [1:nC,1:nV] arrays of Aj are arrays of A, and the others except the main diagonal arrays that are (1), are zero.
    Aj=np.zeros((nC,nC+nV))
    Aj[0:nC,0:nV]=A[0:nC,0:nV]
    j=nV
# Assigning diagonal arrays of y variables.
    for i in range(nC):
        Aj[i,j]=1.0
        j+=1
# !========================End of Construction of Aj
#     print(Aj)
# !############################# Write A ####################################
# !open(unit=2, file='Aj.txt', ACTION="write", STATUS="replace")
# !do row=1,nC
# !     write(2, '(1000F20.7)')( real(Aj(row,col)) ,col=1,nC+nV)
# !end do
# !close(2)
# !==========================================================================

# !-------------------------------------------------Construction of tableauII
    tableauII=np.zeros((nC+1,nC+3))
# Creation of beta in first step (Row: 1:nC, Column: 1:nC)
    tableauII[0:nC,0:nC]=Aj[0:nC,nV:nV+nC]
# -f & -w columns are Zero (Row: 1:nC, Column: nC+2:nC+3)
# Column nC+1 Containing Xs values and in first Step is Blank
# Vector of Constants (b) (Row: 1:nC, Column: nC+3)
    tableauII[0:nC,nC+2]=b[0:nC,0]
# do i=1,nC
# 	tableauII(i,nC+3)=b(i)
# end do
# !Row -f and column -f
# tableauII(nC+1,nC+2)=1.0
    tableauII[nC,nC+1]=1.0
# !Row -w and Column -w
# !tableau(nC+2,nC+3)=1.0
# !Row: -f and Column of Constants Which in First Cycle is 0.0
# !Row: -w and Column of Constants Which in First Cycle is w0
# !=========================================End of Construction of tableauII

# !----------------Construction of Basic Variable Numbers
    bV=np.zeros(nC)
    for i in range(nC):
        bV[i]=nV+i
# do i=1,nC
# 	bV(i)=nV+i
# end do
# !=========End of Construction of Basic Variable Numbers
# !------------------------------Begining of Calculations
# !######################################################
    Cn=1
# !Dar ebteda khodeman Landa ra be onvane voroodi vared mikonim. ya'ni sotoone s dar ebteda khod be khod
# !sotoone marboot be Landa ke sotoone (nC ya 4*L+1) ast entekhab mishavad. pas az an maghadire b/a ra mohasebe
# !mikonim ke har kodam koochektar bood, be onvane satre r entekhab mishavad. va dar natije ozve ars ke bayad amaliate 
# !pivot rooye an anjam shavad moshakhas mishavad. tavajoh shavad ke ellate in ke Landa ra be onvane voroodi entekhab 
# !mikonim in ast ke chenanche Landa dar basis nabashad yaani aslan bargozari soorat nagerefte. Nokteye digari ke bayad be an 
# !tavajoh shavad in ast ke dar hengame mohasebeye b/a, ke dar inja a haman zarayebe landa ast, meghdare a bishtar ya'ni
# !chenanche hameye a'za daraye zarfiate yeksan bashand, meghdare langare ozve bishtar va agar a'za daraye zarfiate yeksan nabashand, 
# !meghdare nesbate langar be zarfiate ozve bishtar. dar natije hamantor ke mibinim bare marboot be ozvi bohranitar mishavad ke daraye 
# !kamtarin nesbate zarfiat be niroo ast va dar natije dar in ozv avalin mafsal tashkil mishavad. 

# !nokteye digari ke bayad be an tavajoh konim in ast ke an radife r ke inja entekhab mishavad darvaghe moteghayere y ast ke bayad
# !az basis kharej shavad, va in moteghaye darvaghe bayangar maghtaei ast ke taslim mishavad. dar natije x e in moteghayer bayad 
# !be onvane moteghayere voroodi dar marhaleye ba'd entekhab shavad. dar natije mibinim ke automaticvar s e marhaleye
# !baad ta'yin shode.
    
    s=nC-1  #dar marhaleye aval Abars darvaghe sotoone Landa ast

    # !--------------------------Entering Landa into Basis
    # !-------------------------------Calculation of Abars
    tableauII[0:nC,nC]=Aj[0:nC,s]
#     tableauII(1:nC,nC+1)=Aj(1:nC,s)
    # !========================End of Calculation of Abars
    minba=0
# !-----------------------------------------Finding Xr
# to have a b/a that has a value we use a loop without any conditions on A and b values.
    for i in range(nC):
        if tableauII[i,nC] > 0.0:
            minba=tableauII[i,nC+2]/tableauII[i,nC]
#-------------------------------------------
    for i in range(nC):
        if tableauII[i,nC] > 0.0:
            ba=tableauII[i,nC+2]/tableauII[i,nC]
            if ba <= minba:
                minba=ba
                ii=i
                r=bV[ii]
# =================================End of Finding Xr
# ----------------------------Pivot Operation on ars
# --------------------------Pivot Operation on row r
    bV[ii]=s
    ars=tableauII[ii,nC]
    
    lowValY = 1e-9
    for i in range(nC+3):
        tableauII[ii,i]=tableauII[ii,i]/ars
    # -------------- Zero out the small values 
    low_values_flags = abs(tableauII) < lowValY  # Where values are low
    tableauII[low_values_flags] = 0  # All low values set to 0
    # ========================================
    
    # !--------------------Pivot Operation on other rows
    for i in range(nC+1):
        if i != ii:
            ais=tableauII[i,nC]
            for j in range(nC+3):
                tableauII[i,j]=tableauII[i,j]-ais*tableauII[ii,j]
     # -------------- Zero out the small values 
    low_values_flags = abs(tableauII) < lowValY # Where values are low
    tableauII[low_values_flags] = 0  # All low values set to 0
    # ========================================
    # ====================================End of Pivoting
    # ==========================End of Entering Landa into Basis
# !ta inja Landa be onvane bare khareji bar saze e'mal shode ast. va Landa varede basis shode ast.

# !agar r=2nc bashad varede do while (r/=nC+nC) namishavim. ya'ni landay be onvane voroodie basis entekhab shode bashad. 
# !dar in halat aslan hich mafsali ijad namishavad. dar vaghe landay bayangare bare ezafi baghi mande bar rooye saze, 
# !ezafe bar bari ast ke sababe tashkile akharin mafsal mishod. va vaghti ke landay az basis kharej mishavad yaani digar
# !bare ezafei baghi namande ke be saze e'mal shavad va dar natije amaliate pivot motevaghef mishavad.

# !haal ke saze bargozari shode, az inja be baad bayad mafsalhaye ijad shode dar saze ra peyda konim.
# !############################# Finding Plastic Hinges
# !####################################################
# !Dar inja yek bar Cbar ra mohasebe mikonim va CA ra mohasebe mikonim va baad az aan varede do while (r/=nC+nC .and. CA == 2 )
# !mishavim. Deghat shavad ke hamin mohasebat ra dar entehaye do while (r/=nC+nC .and. CA == 2 ) va baad az amaliate pivot anjam
# !midahim va dobare Cbarha ra mohasebe mikonim va CA check mishavad va agar masale unbounded shode (CA==1) az do while (r/=nC+nC
# ! .and. CA == 2 ) kharej mishavim va amaliat motevaghef mishavad.
    Cbar=np.zeros(nV+nC)
    Cbar=np.dot(tableauII[nC,0:nC],Aj)+C
    # !Because fo Rounding digits, there might be some error in computations in order of for example 10E-15 and because
    # !sign of Cbar is required, we use below function to set small values of Cbar to zero. 
    # -------------- Zero out the small values 
    low_values_flags = abs(Cbar) < lowValY # Where values are low
    Cbar[low_values_flags] = 0  # All low values set to 0
    # ========================================
    # !==========================End of Computation of New Cbar
    # !------------------------------------------Finding s
    s=r-nC
    s=int(s)
    # !======================================End Finding s
    # Check kardane inke be ezaye hameye C haye manfi, a haye motanazer, manfi ya sefr hastand ya kheyr?
    # !baraye in kar dar ebteda CA ra barabare 1 gharar midahim va dar amaliate zir harja ke motanazer ba C haye manfi a manfi ya sefr nashod
    # !ca ra 2 gharar midahim va az halghe kharej mishavim.
    Cbarasort=np.sort(Cbar)
    indexCa=np.argsort(Cbar)
    # dar ebteda bayad check konim ke aya C morede tahghigh manfi ast ya kheyr. agar manfi bood CA=2 va az halghe kharej mishavim
    # -------------------------------------------Calculation of Abar
#     print('s',s)
#     print('nC',nC)
#     print('Aj[0:nC,indexCa[s]]',Aj[0:nC,indexCa[s]])
#     print('tableauII[0:nV,0:nC]',tableauII[0:nV,0:nC])
    atemp=np.dot(tableauII[0:nV,0:nC],Aj[0:nC,indexCa[s]])
    # -------------- Zero out the small values 
    low_values_flags = abs(atemp) < lowValY # Where values are low
    atemp[low_values_flags] = 0  # All low values set to 0

    # ========================================
    # ===================================End of Calculation of Abars
    CA=1
    i=0
#     print('nC-1',nC-1)
#     print('nC+nV-1',nC+nV-1)
    #-----------------------------Checking if all Abars are non-positive or not
    print("atemp", atemp)
    print("atemp", atemp)

    while CA == 1:
        print(f"atemp[i]={atemp[i]}")
        print(f"bV[i]={bV[i]}", f"nc-1={nC-1}", f"nC+nV-1={nC+nV-1}")
        if atemp[i] > 0 and bV[i] != nC-1 and bV[i] != nC+nV-1:
            CA=2
        elif atemp[i] <= 0 and bV[i] != nC-1 and bV[i] != nC+nV-1:
            CA=1
        i+=1
    #======================End of Checking if all Abars are non-positive or not
    while r != nC+nC-1 and CA == 2:
        #-------------------------------Calculation of Abars
        tableauII[0:nC,nC]=np.dot(tableauII[0:nC,0:nC],Aj[0:nC,s])
        tableauII[nC,nC]=Cbar[s]
        # -------------- Zero out the small values 
        low_values_flags = abs(tableauII[0:nC,nC]) < lowValY # Where values are low
        tableauII[0:nC,nC][low_values_flags] = 0  # All low values set to 0
        # ========================================
        # !-----------------------------------------Finding Xr
        # to have a b/a that has a value we use a loop without any conditions on A and b values.
        for i in range(nC):
            if tableauII[i,nC] > 0.0:
                minba=tableauII[i,nC+2]/tableauII[i,nC]
        #-------------------------------------------
        for i in range(nC):
            if tableauII[i,nC] > 0.0:
                ba=tableauII[i,nC+2]/tableauII[i,nC]
                if ba <= minba:
                    minba=ba
                    ii=i
                    r=bV[ii]
        # =================================End of Finding Xr
        # ----------------------------Pivot Operation on ars
        # --------------------------Pivot Operation on row r
        bV[ii]=s
        ars=tableauII[ii,nC]

        lowValY = 1e-9
        for i in range(nC+3):
            tableauII[ii,i]=tableauII[ii,i]/ars
        # -------------- Zero out the small values 
        low_values_flags = abs(tableauII) < lowValY  # Where values are low
        tableauII[low_values_flags] = 0  # All low values set to 0
        # ========================================

        # !--------------------Pivot Operation on other rows
        for i in range(nC+1):
            if i != ii:
                ais=tableauII[i,nC]
                for j in range(nC+3):
                    tableauII[i,j]=tableauII[i,j]-ais*tableauII[ii,j]
         # -------------- Zero out the small values 
        low_values_flags = abs(tableauII) < lowValY # Where values are low
        tableauII[low_values_flags] = 0  # All low values set to 0
        # ========================================
        # ====================================End of Pivoting
        # !Deghat shavad ke mohasebeye Cbar va CA ra dar entehaye do while (r/=nC+nC .and. CA == 2 ) va baad az amaliate pivot anjam
        # !midahim va dobare Cbarha ra mohasebe mikonim va CA check mishavad va agar masale unbounded shode (CA==1) az do while (r/=nC+nC
        # ! .and. CA == 2 ) kharej mishavim va amaliat motevaghef mishavad.
        Cbar=np.zeros(nV+nC)
        Cbar=np.dot(tableauII[nC,0:nC],Aj)+C
        # !Because fo Rounding digits, there might be some error in computations in order of for example 10E-15 and because
        # !sign of Cbar is required, we use below function to set small values of Cbar to zero. 
        # -------------- Zero out the small values 
        low_values_flags = abs(Cbar) < lowValY # Where values are low
        Cbar[low_values_flags] = 0  # All low values set to 0
        # ========================================
        # !==========================End of Computation of New Cbar
        # !------------------------------------------Finding s
        s=int(r-nC)
        # !======================================End Finding s
        Cbarasort=np.sort(Cbar)
        indexCa=np.argsort(Cbar)
        # dar ebteda bayad check konim ke aya C morede tahghigh manfi ast ya kheyr. agar manfi bood CA=2 va az halghe kharej mishavim
        # -------------------------------------------Calculation of Abar
        
        atemp=np.dot(tableauII[0:nV,0:nC],Aj[0:nC,indexCa[s]])
        # -------------- Zero out the small values 
        low_values_flags = abs(atemp) < lowValY # Where values are low
        atemp[low_values_flags] = 0  # All low values set to 0

        # ========================================
        # ===================================End of Calculation of Abars
        CA=1
        i=0
        #-----------------------------Checking if all Abars are non-positive or not
        while CA == 1:
            print(f"i={i}")
            print(f"atemp[i]={atemp[i]}")
            print(f"bV[i]={bV[i]}", f"nc-1={nC-1}", f"nC+nV-1={nC+nV-1}")
            if atemp[i] > 0 and bV[i] != nC-1 and bV[i] != nC+nV-1:
                CA=2
            elif atemp[i] <= 0 and bV[i] != nC-1 and bV[i] != nC+nV-1:
                CA=1
            i+=1
        #======================End of Checking if all Abars are non-positive or not
    Xn=np.zeros(nC)
        
    for i in range(nC):
        if int(bV[i]) < nC:
            Xn[int(bV[i])]=tableauII[i,nC+2]
            
    print('bV',bV) 
    
    return Xn


Xn=complementarity_programming(nC,nV,A,b,minmax,equ,C)
print('Xn', Xn)
# Computation of ElastoPlastic Responses
## Computation of ElastoPlastic Member Forces
def elastoPlastic_Response(Xn,phi,unitResponse,loadResponse):
    #Xn: is basic variable of mathematical programming that reveals which plastic multipliers are in basis.
    #phi: is the matrix of yeild surfaces of all of the sections that can be plastic.
    #unitResponse: is the response of structure under equilibrated forces due to unit deformataion. unitResponse can be
    #forces or deformations of elements or deflection of nodes that are stored in a list.
    #loadResponse: is the response of structure due to External loadings.
    xbar=np.zeros((len(unitResponse),1))
    xbar=np.dot(phi,np.reshape(Xn[0:-1], (len(Xn)-1, 1)))
    print(unitResponse[0])
    print(unitResponse[1])
    print(unitResponse[2])
    print(unitResponse[3])
    print('-------------------')
    
    for i in range(len(unitResponse)):
        unitResponse[i]=unitResponse[i]*xbar[i][0]
    print('xbar',xbar)
    print(unitResponse[0])
    print(unitResponse[1])
    print(unitResponse[2])
    print(unitResponse[3])
    print('-------------------')
    sumUnit=np.sum(unitResponse, 0)
    EPResponse=sumUnit+loadResponse*Xn[-1]
    print(sumUnit)
    print(EPResponse)
elastoPlastic_Response(Xn,phi,unitMemberForce,Fs)

# print(unitMemberForce[0])
# print(unitMemberForce[1])
# print(unitMemberForce[2])
# print(unitMemberForce[3])
# print('-------------------')
# sumUnit=np.sum(unitMemberForce, 0)
# print(sumUnit)
