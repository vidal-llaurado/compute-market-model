"""Price funded European options by discounted affine-transform inversion.

Fourier-frequency and jump-age quadrature retains stochastic discounting and
Gaussian-mark compound-Poisson jumps. Put-call parity supplies the put price."""
from __future__ import annotations
import numpy as np
from scipy.special import ndtr
from numpy.polynomial.legendre import leggauss


def _coefficients(model, market, t, expiry, delivery, nodes):
    key=('european-transform', market,float(t),float(expiry),float(delivery),int(nodes))
    if key in model._bond_cache: return model._bond_cache[key]
    tau=expiry-t; v,w=model._grid(0,tau)
    ell=model.loading(market,delivery-expiry+v); B=model.rate_B(v)
    state_loading=model.loading(market,delivery-t)
    base=np.log(model.initial(market,delivery))-model.compensator(market,expiry,delivery)
    rate_shift=model.shift_integral(t,expiry)
    def exponent(z):
        z=np.atleast_1d(z)
        c=z[:,None,None]*ell[None,:,:]-B[None,:,:]
        return z*base-rate_shift+model.cumulant(c,'Q')@w
    variance=float(w@np.einsum('ni,ij,nj->n',ell,model.cov,ell))
    # Positive Brownian projection makes the Fourier inversion integrable.
    if variance < 1e-12: raise ValueError('Fourier option needs positive projected diffusion; use payoff Monte Carlo for pure-jump claims')
    ceiling=max(80.,14/np.sqrt(variance))
    z,a=leggauss(nodes);u=(z+1)*ceiling/2;weights=a*ceiling/2
    zero=float(np.real(exponent(0.)[0]));one=float(np.real(exponent(1.)[0]))
    out=(state_loading,model.rate_B(tau),zero,one,u,weights,
         np.exp(exponent(1j*u)-zero),np.exp(exponent(1+1j*u)-one),variance)
    model._bond_cache[key]=out
    return out


def option_price(model,market,t,x,expiry,delivery,strike,put=True,nodes=128):
    """Price per underlying unit, batch-compatible. Clean funded cash at expiry."""
    x=np.asarray(x,float)
    if (x.shape[-1]!=model.dimension or not np.isfinite(x).all()
            or not 0<=t<=expiry<=delivery or strike<=0 or nodes<32):
        raise ValueError('finite states, ordered dates, positive strike and 32+ quadrature nodes required')
    if model.curves[market].kind!='log': raise ValueError('positive compute reference required')
    if expiry==t:
        f=model.value(market,t,x,delivery)
        return np.maximum(strike-f,0.) if put else np.maximum(f-strike,0.)
    if not model.jump_intensity_q:
        v,w=model._grid(0,expiry-t);ell=model.loading(market,delivery-expiry+v);B=model.rate_B(v)
        variance=float(w@np.einsum('ni,ij,nj->n',ell,model.cov,ell))
        correction=-float(w@np.einsum('ni,ij,nj->n',ell,model.cov,B))
        forward=model.value(market,t,x,delivery)*np.exp(correction)
        D=model.discount(t,x,expiry)
        if variance<1e-15:return D*np.maximum(strike-forward if put else forward-strike,0.)
        sd=np.sqrt(variance);d1=np.log(forward/strike)/sd+sd/2;d2=d1-sd
        return D*(strike*ndtr(-d2)-forward*ndtr(-d1)) if put else D*(forward*ndtr(d1)-strike*ndtr(d2))
    ell,B,c0,c1,u,w,phi0,phi1,_=_coefficients(model,market,t,expiry,delivery,nodes)
    phase=np.asarray(x@ell)
    argument=np.exp(1j*(phase[...,None]-np.log(strike))*u)
    p0=.5+np.sum(w*np.imag(argument*phi0)/u,axis=-1)/np.pi
    p1=.5+np.sum(w*np.imag(argument*phi1)/u,axis=-1)/np.pi
    D=np.exp(c0-x@B);DF=np.exp(c1+x@(ell-B))
    call=DF*p1-strike*D*p0
    price=call-DF+strike*D if put else call
    lower=np.maximum(strike*D-DF if put else DF-strike*D,0.)
    upper=strike*D if put else DF
    violation=np.maximum(lower-price,price-upper)
    if np.max(violation)>2e-6*max(1.,float(np.max(upper))):
        if nodes<512:return option_price(model,market,t,x,expiry,delivery,strike,put,nodes*2)
        raise RuntimeError('option Fourier quadrature failed no-arbitrage price bounds')
    return np.minimum(np.maximum(price,lower),upper)


def option_convergence(model,market,t,x,expiry,delivery,strike,put=True):
    p=option_price(model,market,t,x,expiry,delivery,strike,put,128)
    refined=option_price(model,market,t,x,expiry,delivery,strike,put,256)
    return {'price':float(refined),'coarse_price':float(p),
            'quadrature_change':float(refined-p),'method':'affine discounted characteristic transform',
            'frequency_nodes':[128,256]}
