"""Check of Hansen 2010 Table II and Eq. (23).
Purpose: (i) compute 1/K and Hansen's length l = sqrt((zeta0+zetar)/eta0) from Table II;
(ii) solve Eq. (22) for the slit with wall condition u = Omega = 0 using sympy and compare the flow-rate ratio Q with Eq. (23).
All numbers are reported by this script only."""
import sympy as sp, math
# (i) Table II (T, eta0, etar [1e-3 Pa s], zeta0, zetar [1e-21 kg m/s])
T2=[(284,.92,.17,1.4,1.6),(289,.83,.17,1.2,1.5),(293,.76,.17,1.2,1.4),(299,.7,.17,.9,1.2),
    (305,.61,.16,.93,1.2),(309,.56,.164,.83,1.0),(313,.54,.161,.8,.9),(319,.48,.160,.7,.92)]
print("T eta_r/eta0  1/K[nm]  l=sqrt((z0+zr)/eta0)[nm]")
for T,e0,er,z0,zr in T2:
    e0*=1e-3; er*=1e-3; z=(z0+zr)*1e-21
    K=math.sqrt(4*er*e0/((e0+er)*z)); l=math.sqrt(z/e0)
    print(T, round(er/e0,3), round(1e9/K,3), round(1e9*l,3))
# Fig. 3 parameters
e0,er,z=1e-3,1.7e-4,(1.1+1.4)*1e-21
print("Fig3 1/K nm",1e9/math.sqrt(4*er*e0/((e0+er)*z)))
# (ii) sympy: slit, Eq. (22) steady, u(+-h)=Omega(+-h)=0.
# W=Omega_y. First integral of the first equation (symmetry, c=0): (e0+er)u' - 2 er W = -G z ; second: 2 er(u'-2W)+zeta W''=0
# => W'' = K^2 W + ..., solution W = A sinh(Kz) - G z/(2 e0), A = G h/(2 e0 sinh(Kh))
import sympy as sp
z,h,G,e0s,ers,zs=sp.symbols('z h G eta0 etar zeta',positive=True)
K=sp.sqrt(4*ers*e0s/((e0s+ers)*zs))
A=G*h/(2*e0s*sp.sinh(K*h))
W=A*sp.sinh(K*z)-G*z/(2*e0s)
up=2*W-zs*sp.diff(W,z,2)/(2*ers)
# check: residuals of the 1st and 2nd equation vanish
r1=sp.simplify(((e0s+ers)*up-2*ers*W+G*z).rewrite(sp.exp))
r2=sp.simplify((2*ers*(up-2*W)+zs*sp.diff(W,z,2)).rewrite(sp.exp))
print("residual1,residual2:",r1,r2)
Q=sp.simplify(-sp.integrate(z*up,(z,-h,h)))
Q0=2*G*h**3/(3*e0s)          # eta_r=0: viscosity eta0
X_hansen=-3*ers*(sp.tanh(K*h)-K*h)/(sp.tanh(K*h)*(e0s+ers)*(K*h)**2)
vals={e0s:1e-3,ers:1.7e-4,zs:2.5e-21,h:50e-9,G:1.0}
print("Q/Q0 (sympy)      :",sp.N((Q/Q0).subs(vals)))
print("1 - X_Hansen(23)  :",sp.N((1-X_hansen).subs(vals)))
print("Q/Q0 alt: 1/(1+..) ",sp.N((Q0/Q0).subs(vals)))
d=sp.simplify((Q/Q0-(1-X_hansen)).rewrite(sp.exp))
print("difference (simplify):",d)
print("numerical difference, 3 points:",[sp.N((Q/Q0-(1-X_hansen)).subs({**vals,h:hh*1e-9})) for hh in (3,20,100)])
# Fig. 3 value: h = 10 nm (channel height 20 nm) -> %?
for H in (20,100,200):
    v={**vals,h:H/2*1e-9}; print("channel height",H,"nm: reduction % (23) =",sp.N(100*X_hansen.subs(v)))

# (iii) Kolpashchikov 1983 Eq. (2.9) [slit] and Migun 1981 Eq. (5) [tube] against Hansen Eq. (23) / Chaturani-Mahajan 1982 Eq. (9)
# mapping: kappa=2 eta_r, mu = eta0 - eta_r (Eringen mu), gamma = zeta ; alpha=0 (nu=0 at the wall)
import math
from scipy.special import iv
e0,er,zt,hh=1.0e-3,0.17e-3,2.5e-21,30e-9
kap=2*er; mu=e0-er; gam=zt; al=0.0
mun=mu+kap/2
delta=2*kap*(1-al)/(2*(mu+kap)-kap*al)
kh=hh*math.sqrt((2*mu+kap)*kap/((mu+kap)*gam))
Q_kol=1+3*delta/(2*kh**2)*(1-kh/math.tanh(kh))
Kh=hh*math.sqrt(4*er*e0/((e0+er)*zt))
Q_han=1-3*er*(Kh/math.tanh(Kh)-1)/((e0+er)*Kh**2)
print("slit  Kolpashchikov(2.9):",Q_kol," Hansen(23):",Q_han," mun==eta0:",abs(mun-e0)<1e-18)
# tube: Migun (5) and Chaturani-Mahajan (9); general value such as alpha=0.3
for al in (0.0,0.3,0.7):
    r0=40e-9
    kM=r0*math.sqrt((2*mu+kap)*kap/((mu+kap)*gam))
    dM=2*kap*(1-al)/(2*mu+kap*(2-al))     # delta as published by Migun
    Qm=1+4*dM/kM**2*(1-kM*iv(0,kM)/(2*iv(1,kM)))
    k=kap/2; mus=mu; s=2*(1-al)/al if al>0 else float('inf')   # Chaturani: dv/dr=-(2+s)w, Migun: nu=(al/2) rot v
    mu_c=mus+k
    sbar=(1/(1+2*mu_c/(s*(mu_c+k)))) if al>0 else 1.0
    l2=gam*(1/k+1/mu_c)/4; a=r0/math.sqrt(l2)
    T=2/a**2-iv(0,a)/(a*iv(1,a))
    Qc=1+4*sbar*k*T/(mus+2*k)
    print(f"tube alpha={al}: Migun(5) Q/Qn={Qm:.6f}  Chaturani(9) Q/Qn={Qc:.6f}  k_Migun={kM:.4f} a_Chat={a:.4f}")
