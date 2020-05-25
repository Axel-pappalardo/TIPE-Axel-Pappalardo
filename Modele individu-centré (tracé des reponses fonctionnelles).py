'''
Ce code est une simulation individu-centré d'un banc de poisson avec des predateurs (qui eux ne forment pas de banc).
Il calcule et affiche le nombre de proies devorés en moyenne en fonction du nombre de proies
il lance donc la simulation pour different nombre de proies (ici de 5 a 25 de 5 en 5) et effectue la moyenne sur nbr_iter iterations (ici 5) la duré de calcul est rapidement tres élévé (une moyenne sur 25 iterations pour un nombre de proie allant de 5 a 100 de 5 en 5 est estimé a 48h environ avec mon ordinateur(estimation basse))
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

'''on veut creer des poissons fictifs a partir du poisson 1 hors du carré [au coord (x+L,y),(x-L,y),(x,y+L),...,(x+L,y+L),(x+L,y-L)] et on prend le min des distances trouvé comme distance reel'''

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
    return res

#ici les poissons on un espace vital de R, un rayon de repoussement de Re un rayon d'attraction de Ra et un rayon d'alignement Ral (il percoivent donc dans un cercle de Ra autour d'eux)

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
Va = 5

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


def mort_proie(L_m,L_p):    #a la fin de chaque dt on vérifie les distances proies/predateurs et on tue les proies trop proche d'un predateur
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
        L_last_coord.append(last_coord)   #on récupère ici les dernières coordonnées des poissons tué
    L_p = np.array(L_p)
    return L_p,L_last_coord
    
    
##Test
t1 = 50
dt = 0.1
L = 50      #taille du carré dans lequel les poissons évoluent
nbr = 5     #nombre de predateurs
tm = 10     #temps a partir duquel on met des predateurs


ti = time.clock()
L_N = [i*5 for i in range(1,5)]
L_f = []

nbr_iter = 5

for k in L_N:
    N = k
    S_mort = 0
    for i in range(nbr_iter):
        t0 = 0
        ()
        L_m = []
        L_p = placement2D(N,0.1,L)
        L_p = Dir_ini(L_p)
        L_p = np.array(L_p)
        
        New_L_p = L_p
        
        while t0<t1:
            New_L_p = New_temps(New_L_p,dt)
            New_L_p = np.array(New_L_p)
            if t0 >= tm:
                if L_m == []:
                    L_m = placement2D(nbr,0.5,L)
                    L_m = Ini_satiete(L_m)
                    L_m = np.array(L_m)
                else:
                    L_m = New_pos_predateur(L_m)
                L_m = np.array(L_m)
                New_L_p,L_last_coord = mort_proie(L_m,New_L_p)
            t0 = t0 + dt
        nbr_mort = len(L_p) - len(New_L_p)
        S_mort += nbr_mort
    moy_mort = S_mort/nbr_iter    
    L_f.append(moy_mort)      
    print('k = ' + str(k) + ' done')

tf = time.clock()
deltat = (tf-ti)/60
deltat = round(deltat,5)

'''
on mets dans un fichier text les resultats (nombre de proies tuer pour le nombre de proies presente) dans des fichier txt pour pouvoir les conserver et les reutiliser dans un code specifique a l'affichage des résultats
'''

fichier = open("banc_poisson_fonctionnelle_7-1.txt","w")
fichier.write('L_N , L_f')
fichier.write('\n')
for i in range(len(L_N)):
    fichier.write(str(L_N[i]))
    fichier.write(' , ')
    fichier.write(str(L_f[i]))
    fichier.write(str('\n'))
fichier.write('tps = ' + str(deltat))
fichier.close()

plt.plot(L_N,L_f,label = 'reponse fonctionelle')

plt.title('tps = ' + str(deltat))
plt.show()
