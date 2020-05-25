'''
Ce code est une simulation individu-centré d'un banc de poisson avec des predateurs (qui eux ne forment pas de banc).
Il affiche le parcours des poissons sur un graphique avec:
points bleu: position initiales des proies
fleche rouge: positions finales des proies (pointe dans leur direction finale
pointillés jaune: trajectoire des proies
cercle violet: rayon de repulsion autour des positions finales des proies
points rouges: positions initiale des predateurs
points violets: positions finale des predateurs
pointillés violet: trajectoire des predateurs
croix noir: lieu ou une proie c'est faite tuer par un predateur 
'''

##Importation des modules
import numpy as np
from math import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import operator
import random as rd
import numpy.random as nrd

##Code Placement (et definition de Distance)

'''on veut creer des poissons fictifs a partir du poisson 1 hors du carré [au coord (x+L,y),(x-L,y),(x,y+L),...,(x+L,y+L),(x+L,y-L)] et on prend le min des distances trouvé comme distance réel'''

def Distance_Tore(point1,point2,L):
    x,y = point1[0],point1[1]
    L_point_fic = [[x,y],[x+L,y],[x-L,y],[x,y+L],[x,y-L],[x-L,y-L],[x+L,y-L],[x+L,y+L],[x-L,y+L]]
    L_dist = []
    for point_fic in L_point_fic:
        dist_fic = Distance(point_fic,point2)
        L_dist.append(dist_fic)
    dist = min(L_dist)
    return dist

def Distance_Torebis(point1,point2,L):
    x,y = point1[0],point1[1]
    L_point_fic = [[x,y],[x+L,y],[x-L,y],[x,y+L],[x,y-L],[x-L,y-L],[x+L,y-L],[x+L,y+L],[x-L,y+L]]
    L_dist = []
    for point_fic in L_point_fic:
        dist_fic = Distance(point_fic,point2)
        L_dist.append(dist_fic)
    dist = min(L_dist)
    ind = L_dist.index(dist)
    point_fic = L_point_fic[ind]
    return dist,point_fic
    
def Distance(point1,point2):
    dim = 2
    dist = 0
    for k in range(dim):
        dk = abs(point1[k]-point2[k])
        dist = dist + dk**2
    return sqrt(dist)

def placement2D(N,R,L):   #N=nbr de poissons R='rayon' du poisson et L=coté du carré dans lequel evoluent les poissons
    max = 0
    def possible(c,res):
        for k in res:
            if Distance(k,c) < 2*R:
                return False
        return True
        
    res = [R + np.random.rand(2)*(L-2*R)]
    
    while len(res) < N:
        p = R + np.random.rand(2)*(L-2*R)
        if possible(p,res):
            res.append(p)
        else:
            res = [R + np.random.rand(2)*(L-2*R)]
        if len(res) > max:
            max = len(res)
            print(max)   #J'affiche ici combien de poissons la fonction a placé au mieux jusque là
    return res



#ici les poissons on un espace vital de R, un rayon de repoussement de Re un rayon d'attraction de Ra et un rayon d'alignement Ral (il percoivent donc dans un cercle de Ra autour d'eux)


##Code cercle

def f_Cercle(L_p,r,couleur):
    for k in range(len(L_p[:,0])):
    
        x,y = L_p[k][0:2]
    
        theta = np.linspace(0,2*pi,100)

        Rx = r*np.cos(theta) + x
        Ry = r*np.sin(theta) + y

        plt.plot(Rx,Ry,couleur)

##Code Liste Poisson

'L_poi = [num poisson,coord_x,coord_y,direction]'
def f_Lpoi(L_p):
    L_poi = []
    for i in range(len(L_p[:,0])):
        L_loc = [i,L_p[i][0],L_p[i][1]]
        L_poi.append(L_loc)
    L_poi = np.array(L_poi)
    return L_poi

##Délimitation de l'espace

def f_Teleportation(px,py,L):
    if px > L:
        px = 0
    elif px < 0:
        px = L
    if py > L:
        py = 0
    elif py < 0:
        py = L
    return px,py
    
##Code initialisation direction

def Dir_ini(L_p):
    npLp = np.array(L_p)
    L_p = [list(i) for i in L_p]
    for k in range(len(npLp[:,0])):
        angle = rd.random()*2*pi
        L_p[k].append(angle)
    return L_p


##Code alignement
Ral = 5     #Rayon d'alignement
Va = 5      #Vitesse d'alignement

def f_prox_align(p,L_p):        #cette fonction recupere la liste des poissons avec lequels le poisson s'aligne
    L_poi = f_Lpoi(L_p)
    L_dist = [[Distance_Tore(p,L_p[k][0:2],L),k] for k in range(len(L_p[:,0]))]
    L_dist = sorted(L_dist, key=operator.itemgetter(0))
    L_indprox = []
    for i in range(len(L_dist)):
        if L_dist[i][0] <= Ral and L_dist[i][0] >= Rr:
            L_indprox.append(L_dist[i][1])
    L_align = np.array([L_poi[k] for k in L_indprox])
    return L_align
  

def angle_prox(p,L_align):
    L_angleprox = [L_p[k,2] for k in range(len(L_align))]
    return L_angleprox

def f_Moy(L):
    somme = 0
    taille = len(L)
    for k in range(taille):
        somme += L[k]
    if L == []:
        return None
    moy = somme/taille
    return moy

def New_dir(p,L_p):                       
    L_align = f_prox_align(p,L_p)
    L_angleprox = angle_prox(p,L_align)
    if f_Moy(L_angleprox) == None:
        New_dir = p[2] + (nrd.randn())/4
    else:
        New_dir = f_Moy(L_angleprox) + nrd.randn()/4
    return New_dir

def V_al(p,L_p):
    dir = p[2]
    Vx = Va*cos(dir)
    Vy = Va*sin(dir)
    return Vx,Vy


##Code mouvement

k = 5
Rr = 1      #Rayon de repoussement
Ra = 10     #Rayon d'attraction
a = 10
b = 0.03
alpha = 1

def f_prox(p,L_p):      
    L_poi = f_Lpoi(L_p)
    L_dist = [[Distance_Tore(p,L_p[k][0:2],L),k] for k in range(len(L_p[:,0]))]
    L_dist = sorted(L_dist, key=operator.itemgetter(0))
    L_indprox = []
    j = 0
    while L_dist[j][0] <= Ra:
        L_indprox.append(L_dist[j][1])
        j += 1
    L_indprox.remove(L_indprox[0])
    L_proche = np.array([L_poi[k] for k in L_indprox])
    return L_proche

#on prend pour calculer l'angle entre p1 et p2 les coordonnées du poisson fictif obtenue a partir de p1 le plus proche de p2
def f_angle(p1,p2):
    d,p_fic = Distance_Torebis(p1,p2,L)
    cteta = (p2[0]-p_fic[0])/d
    teta = acos(cteta)
    steta = (p2[1]-p_fic[1])/d
    if asin(steta) < 0:
        teta = -teta
    return teta

def Vx_i(pi,pj):
    theta = f_angle(pi,pj)
    d = Distance_Tore(pi,pj,L)
    if d == 0:
        return 0
    if d < Rr:
        Vx = -((k/d)**alpha)*cos(theta)
        if not -3<Vx<3:
            if Vx<0:
                Vx = -3
            else:
                Vx = 3
    elif Ral < d <=Ra:
        Vx = a*exp(-b*d)*cos(theta)
    else:
        Vx = 0
    return Vx

def Vy_i(pi,pj):
    theta = f_angle(pi,pj)
    d = Distance_Tore(pi,pj,L)
    if d == 0:
        return 0
    if d < Rr:
        Vy = -((k/d)**alpha)*sin(theta)
        if not -3<Vy<3:
            if Vy<0:
                Vy = -3
            else:
                Vy = 3
    elif Ral < d <=Ra:
        Vy = a*exp(-b*d)*sin(theta)
    else:
        Vy = 0
    return Vy
  
def f_Lv(p,L_p):        #on récupère TOUTES les vitesses d'un poisson dans une liste
    Valx,Valy = V_al(p,L_p)
    L_Vx = []
    L_Vy = []
    for k in range(len(L_p)):
        Vx_k = Vx_i(p,L_p[k,0:2])
        Vy_k = Vy_i(p,L_p[k,0:2])
        L_Vx.append(Vx_k)
        L_Vy.append(Vy_k)
    L_Vx.append(Valx)
    L_Vx.append(rd.random()*2 - 1)
    L_Vy.append(Valy)
    L_Vy.append(rd.random()*2 - 1)
    for i in range(len(L_m)):
        L_Vx.append(f_Vxproie_i(p,L_m[i]))
        L_Vy.append(f_Vyproie_i(p,L_m[i]))
    return L_Vx,L_Vy

 
def New_pos_ang(p,L_p,dt):
    [L_Vx,L_Vy] = f_Lv(p,L_p)
    New_x,New_y = p[0],p[1]
    p[2] = New_dir(p,L_p)
    p[2] = p[2]%(2*pi)
    New_ang = f_dir_fuite_pre(p,L_m)
    for k in range(len(L_Vx)):
        New_x = New_x + dt*(L_Vx[k])
        New_y = New_y + dt*(L_Vy[k])
    New_x,New_y = f_Teleportation(New_x,New_y,L)
    return [New_x,New_y,New_ang]


def New_temps(L_p,dt):
    New_L_p = []
    for k in range(len(L_p)):
        New_p = New_pos_ang(L_p[k],L_p,dt)
        New_L_p.append(New_p)
    return New_L_p


##Code predateur
l = 2
z = 0.025
Vm = 1.5
w = 1.0
u = 0.01
Rapre = 15        #Rayon attraction predateur
Rrep_pre = 0.75   #Rayon de repulsion entre predateur
Ral_fuite = 10    #Rayon d'alignement a l'opposé du predateur
Portée = 0.5      #Rayon de capture du predateur

def Ini_satiete(L_m):
    for k in range(nbr):
        L_m[k] = list(L_m[k])
        L_m[k].append(0)
    return L_m

def f_Vxproie_i(pi,prej):
    theta = f_angle(pi,prej)
    d = Distance_Tore(pi,prej,L)
    if d <= Ra:
        Vx = -l*exp(-z*d)*cos(theta)
    else:
        Vx = 0
    return Vx
    
def f_Vyproie_i(pi,prej):
    theta = f_angle(pi,prej)
    d = Distance_Tore(pi,prej,L)
    if d <= Ra:
        Vy = -l*exp(-z*d)*sin(theta)
    else:
        Vy = 0
    return Vy

def f_dir_fuite_pre(p,L_m):
    L_theta = []
    for pred in L_m:
        if Distance_Tore(p,pred,L) <= Ral_fuite:
            thetak = f_angle(p,pred)
            L_theta.append(thetak)
    theta_fuite = f_Moy(L_theta)
    if theta_fuite == None:
        return p[2]
    else:
        return (theta_fuite + pi)%(2*pi)

#on utilise placement2D pour initialiser les predateurs

def f_Vxpre_i(prei,pj):
    theta = f_angle(prei,pj)
    d = Distance_Tore(prei,pj,L)
    if d <= Rapre:
        Vx = w*exp(-u*d)*cos(theta)
    else:
        Vx = 0
    return Vx
    
def f_Vypre_i(prei,pj):
    theta = f_angle(prei,pj)
    d = Distance_Tore(prei,pj,L)
    if d <= Rapre:
        Vy = w*exp(-u*d)*sin(theta)
    else:
        Vy = 0
    return Vy
    
def Vx_eloi_pre(prei,prej):
    theta = f_angle(prei,prej)
    d = Distance_Tore(prei,prej,L)
    if d == 0:
        return 0
    if d < Rrep_pre:
        Vx = -((k/d)**alpha)*cos(theta)
        if not -3<Vx<3:
            if Vx<0:
                Vx = -3
            else:
                Vx = 3
    else:
        Vx = 0
    return Vx
    
def Vy_eloi_pre(prei,prej):
    theta = f_angle(prei,prej)
    d = Distance_Tore(prei,prej,L)
    if d == 0:
        return 0
    if d < Rrep_pre:
        Vy = -((k/d)**alpha)*sin(theta)
        if not -3<Vy<3:
            if Vy<0:
                Vy = -3
            else:
                Vy = 3
    else:
        Vy = 0
    return Vy
    
def f_Lvpre(m,L_p):
    L_Vmx = []
    L_Vmy = []
    if m[2] == 0:
        for k in range(len(L_p)):
            Vmxk = f_Vxpre_i(m,L_p[k])
            Vmyk = f_Vypre_i(m,L_p[k])
            L_Vmx.append(Vmxk)
            L_Vmy.append(Vmyk)
    for k in range(len(L_m)):
        L_Vmx.append(Vx_eloi_pre(m,L_m[k]))
        L_Vmy.append(Vy_eloi_pre(m,L_m[k]))
    return L_Vmx,L_Vmy

def New_pos_predateur(L_m):
    New_L_m = []
    for k in range(len(L_m)):
        mk = L_m[k]
        mx,my,msat = mk[0],mk[1],mk[2]
        L_Vmx,L_Vmy = f_Lvpre(mk,New_L_p)
        New_mx,New_my,New_msat = mx,my,msat
        for k in range(len(L_Vmx)):
            Vx = L_Vmx[k]
            Vy = L_Vmy[k]
            New_mx += Vx*dt
            New_my += Vy*dt
        New_mx,New_my = f_Teleportation(New_mx,New_my,L)
        New_mk = [New_mx,New_my,New_msat]
        New_L_m.append([New_mk[0],New_mk[1],New_mk[2]])
    return New_L_m


def mort_proie(L_m,L_p):    #a la fin de chaque dt on verifie les distances proies/predateurs et on tue les proies trop proche d'un predateur.
    L_ind = []
    for i in range(len(L_m[:,0])):
        if L_m[i,2] == 0:
            for k in range(len(L_p[:,0])):
                d_p_pre = Distance_Tore(L_m[i],L_p[k],L)
                if d_p_pre < Portée :
                    L_ind.append(k)
                    L_m[i,2] = 100
        else:
            L_m[i,2] -= 1
    j = 0
    L_last_coord = []
    L_p = list(L_p)
    for k in L_ind:
        r = k-j
        last_coord = L_p.pop(r)
        j += 1
        L_last_coord.append(last_coord) #on récupère ici les dernières coordonnées des poissons tué
    L_p = np.array(L_p)
    return L_p,L_last_coord
    
    
##Test
t0 = 0
t1 = 50     
tm = 10     #temps a partir duquel on met des predateurs
N = 25      #nombre de proies
dt = 0.1
L = 50      #taille du carré dans lequel les poissons évoluent
nbr = 5     #nombre de predateurs

ti = time.clock()

L_m = []

L_p = placement2D(N,0.1,L)
L_p = Dir_ini(L_p)
L_p = np.array(L_p)
L_x = L_p[:,0]
L_y = L_p[:,1]

plt.figure(1)
plt.plot(L_x,L_y,'o')
plt.xlim(0,25)
plt.ylim(0,25)

New_L_p = L_p

L_trajx = [ [] for i in range(N) ]
L_trajy = [ [] for i in range(N) ]
L_dir = [ [] for i in range(N) ]

L_trajpredx = [ [] for i in range(nbr)]
L_trajpredy = [ [] for i in range(nbr)]

L_L_last_coord = []

for k in range(N):
    L_trajx[k].append(L_p[k][0])
    L_trajy[k].append(L_p[k][1])


while t0<t1:
    New_L_p = New_temps(New_L_p,dt)
    
    New_L_p = np.array(New_L_p)
    L_xx = New_L_p[:,0]
    L_yy = New_L_p[:,1]
    L_theta = New_L_p[:,2]
    for k in range(len(L_xx)):
        L_trajx[k].append(L_xx[k])
        L_trajy[k].append(L_yy[k])
        L_dir[k].append(L_theta[k])
    if t0 >= tm:
        if L_m == []:
            L_m = placement2D(nbr,0.5,L)
            L_m = Ini_satiete(L_m)
            L_m = np.array(L_m)
            L_mx = L_m[:,0]
            L_my = L_m[:,1]
            plt.plot(L_mx,L_my,'or')
            for k in range(nbr):
                L_trajpredx[k].append(L_m[k][0])
                L_trajpredy[k].append(L_m[k][1])
        else:
            L_m = New_pos_predateur(L_m)
            for k in range(nbr):
                L_trajpredx[k].append(L_m[k][0])
                L_trajpredy[k].append(L_m[k][1])
        L_m = np.array(L_m)
        New_L_p,L_last_coord = mort_proie(L_m,New_L_p)
        L_L_last_coord.append(np.array(L_last_coord))
        if not L_last_coord == []:
            L_last_coord = np.array(L_last_coord)
            L_last_coordx = L_last_coord[:,0]
            L_last_coordy = L_last_coord[:,1]
            plt.plot(L_last_coordx,L_last_coordy,'+k')
    t0 = t0 + dt

#on trace ici la trajectoire des proies
for k in range(len(L_trajx)):
    plt.plot(L_trajx[k],L_trajy[k],'y,')
plt.plot(L_trajx[0],L_trajy[0],'r,')    #pour tracer une trajectoire en rouge

#on trace ici la trajectoire des prédateurs
for k in range(len(L_trajpredx)):
    plt.plot(L_trajpredx[k],L_trajpredy[k],'m,')
    
tf = time.clock()
deltat = tf-ti
deltat = round(deltat,5)

L_m = np.array(L_m)
f_Cercle(New_L_p,Rr,'m')

#on trace ici les positions finals des poissons sous la forme de fleche (pointant dans la direction des poissons)
for k in range(len(L_xx)):
    tetak = New_L_p[k][2]
    x = L_xx[k]
    y = L_yy[k]
    plt.axes().arrow(x, y, cos(tetak)*0.01,sin(tetak)*0.01, head_width=0.5, head_length=0.5,fc='r')

#on trace les positions finales des predateurs
L_mx = L_m[:,0]
L_my = L_m[:,1]
plt.plot(L_mx,L_my,'om')

'''on mets ici toutes coordonnées et directions des différents poissons dans un fichier txt pour pouvoir les récupérer et les réutiliser dans un code spécifique pour l'affichage des résultats'''

fichier = open("banc_poisson_traj_7-2.txt","w")
fichier.write('L_trajx:')
fichier.write('\n')
for i in range(len(L_trajx)):
    for k in range(len(L_trajx[i])):
        fichier.write(str(L_trajx[i][k]))
        fichier.write(' , ')
    fichier.write(str('\n'))

fichier.write('L_trajy:')
fichier.write('\n')
for i in range(len(L_trajy)):
    for k in range(len(L_trajy[i])):
        fichier.write(str(L_trajy[i][k]))
        fichier.write(' , ')
    fichier.write(str('\n'))

fichier.write('L_dir:')
fichier.write('\n')
for i in range(len(L_dir)):
    for k in range(len(L_dir[i])):
        fichier.write(str(L_dir[i][k]))
        fichier.write(' , ')
    fichier.write(str('\n'))

fichier.write('L_trajpredx:')
fichier.write('\n')    
for i in range(len(L_trajpredx)):
    for k in range(len(L_trajpredx[i])):
        fichier.write(str(L_trajpredx[i][k]))
        fichier.write(' , ')
    fichier.write(str('\n'))

fichier.write('L_trajpredy:')
fichier.write('\n')
for i in range(len(L_trajpredy)):
    for k in range(len(L_trajpredy[i])):
        fichier.write(str(L_trajpredy[i][k]))
        fichier.write(' , ')
    fichier.write(str('\n'))

fichier.write('L_L_last_coord:')
fichier.write('\n')
for i in range(len(L_L_last_coord)):
    fichier.write(str(L_L_last_coord[i]))
    fichier.write(' - ')
fichier.write('\n')
fichier.write('tps = ' + str(deltat))
fichier.close()

plt.title("tps=" + str(deltat))
plt.axis("equal")
plt.show()
