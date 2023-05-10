import awkward as ak
import uproot
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.nanoevents.methods.base import NanoEventsArray



def muon(sample,cut_name,tag):
    if cut_name == 'num':
        pass_mu = (sample.Muon.pt > 10) & (sample.Muon.looseId) & (sample.Muon.pfIsoId >= 2)
        pass_pho = (sample.Photon.pt > 10)
        return(sample[(ak.num(pass_pho,axis=1)==1) & (ak.num(pass_mu,axis=1)==2)])
    if cut_name == 'trigger':
        if tag == 'IsoMu24':
            return sample[sample.HLT.IsoMu24]
        if tag == 'double':
            return sample[(sample.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL) | (sample.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ) | (sample.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8) | (sample.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8)]
    if cut_name == 'cutBasedID':
        if tag == 'tight':
            return (sample.Muon[sample.Muon.tightId],sample)
    if cut_name == 'pf':
        if tag == 'tight':
            return (sample[sample.pfIsoId >= 4])
#            return (sample[sample.pfRelIso04_all < 0.15])
        if tag == 'medium':
            return(sample[sample.pfIsoId >= 3])
    if cut_name == 'eta':
        if tag == '2.4':
            return sample[abs(sample.eta) < 2.4]
    if cut_name == 'pt':
            pt = sample.pt
            filtered_pt = [(ak.num(pt) >= 2) & (ak.min(pt[:, :2], axis=1) >= 20) & (ak.max(pt[:, :2], axis=1) >= 25)]
            return(tag[filtered_pt[0]])
    if cut_name == 'pid':
        pid_cut = (ak.sum(sample.Muon.pdgId,axis=1)==0) & (ak.num(sample.Muon.pdgId,axis=1)==2)
        filled_pid_cut = ak.where(ak.is_none(pid_cut), False, pid_cut)
        return sample[filled_pid_cut]
    if cut_name == 'mass':
        return(sample[tag >= 50])

        
def photon(sample,cut_name,tag):
    if cut_name == 'pt':
        if tag == '10':
            pt_cut = sample.Photon[sample.Photon.pt > 10]
            return sample[(ak.num(pt_cut) != 0)]
        if tag == '35':
            pt_cut = sample.Photon[sample.Photon.pt > 35]
            return sample[(ak.num(pt_cut) != 0)]
    if cut_name == 'eta':
        if tag =='2.5':
            pt_cut = sample.Photon[(abs(sample.Photon.eta) < 2.5) & ~((abs(sample.Photon.eta) > 1.4442) & (abs(sample.Photon.eta) < 1.566))]
            return sample[(ak.num(pt_cut) != 0)] 
    if cut_name == 'id':
        if tag == 'tight':
            pt_id = (sample.Photon.vidNestedWPBitmap[:] & 1) + 2*((sample.Photon.vidNestedWPBitmap[:] >> 1) & 1) 
            scEta_id = (sample.Photon.vidNestedWPBitmap[:] >> 2 & 1) + 2*((sample.Photon.vidNestedWPBitmap[:] >> 3) & 1) 
            HoE_id = (sample.Photon.vidNestedWPBitmap[:] >> 4 & 1) + 2*((sample.Photon.vidNestedWPBitmap[:] >> 5) & 1) 
            siesie_id = (sample.Photon.vidNestedWPBitmap[:] >> 6 & 1) + 2*((sample.Photon.vidNestedWPBitmap[:] >> 7) & 1) 
            chiso_id = (sample.Photon.vidNestedWPBitmap[:] >> 8 & 1) + 2*((sample.Photon.vidNestedWPBitmap[:] >> 9) & 1) 
            neuiso_id = (sample.Photon.vidNestedWPBitmap[:] >> 10 & 1) + 2*((sample.Photon.vidNestedWPBitmap[:] >> 11) & 1) 
            phoiso_id = (sample.Photon.vidNestedWPBitmap[:] >> 12 & 1) + 2*((sample.Photon.vidNestedWPBitmap[:] >> 13) & 1)
            return sample[(ak.any(pt_id ==3, axis = 1)) &  (ak.any(scEta_id ==3, axis = 1)) & (ak.any(HoE_id ==3, axis = 1)) & (ak.any(siesie_id ==3, axis = 1)) & (ak.any(chiso_id ==3, axis = 1)) & (ak.any(neuiso_id ==3, axis = 1)) & (ak.any(phoiso_id ==3, axis = 1))]
    if cut_name == 'prompt':
        photon_prompt = sample.Photon.genPartFlav[sample. Photon.genPartFlav== 1]
        return sample[(ak.all(photon_prompt == 1, axis=1))]
    if cut_name == 'single':
        return(sample[ak.num(sample.Photon)==1])

def VARIABLES(samples,mass,key,event_before):
    VARIABLES = {
            'photon_pt':samples.Photon.pt[(samples.Photon.eta < 2.5) & (samples.Photon.pt > 10)],
            'photon_eta':samples.Photon.eta[(samples.Photon.eta < 2.5) & (samples.Photon.pt > 10)],
            'muon1_pt':samples.Muon[:,0].pt,
            'muon2_pt':samples.Muon[:,1].pt,
            'muon1_eta':samples.Muon[:,0].eta,
            'muon2_eta':samples.Muon[:,1].eta,
            'muon_mass':mass,
            'npvsGood':samples.PV.npvsGood,
            'Rho_Calo':samples.Rho.fixedGridRhoFastjetCentralCalo,
            'Rho_tracker':samples.Rho.fixedGridRhoFastjetCentralChargedPileUp,
            }
    # if key != 'zg':
    #     PU = {
    #         'npvsGood':samples.PV.npvsGood,
    #         'Rho_Calo':samples.Rho.fixedGridRhoFastjetCentralCalo,
    #         'Rho_tracker':samples.Rho.fixedGridRhoFastjetCentralChargedPileUp,
    #     }
    #     VARIABLES.update(PU)
    if key != 'data':
        dict_genweight = {'generator_weight':np.sign(samples.Generator.weight)}
        event_number = event_before
        nPU = {'nPU':samples.Pileup.nPU}
        VARIABLES.update(dict_genweight)
        VARIABLES.update(event_number)
        VARIABLES.update(nPU)
    return(VARIABLES)

def MASS(samples):
    tem_Muon = samples.Muon[ak.num(samples.Muon.tightId) == 2]
    muon_vectors = ak.zip(
    {
        "pt": tem_Muon["pt"],
        "eta": tem_Muon["eta"],
        "phi": tem_Muon["phi"],
        "mass": tem_Muon["mass"],
    },
    with_name="PtEtaPhiMLorentzVector",
    )
    invariant_mass = (muon_vectors[:, 0] + muon_vectors[:, 1]).mass
    return(invariant_mass)

def EVENT_NUM(samples):
#    NUM = {
#        "event_num":len(samples)
#    }
    return(len(samples))

def DR(obj_A, obj_B, drmin=0.3):
    ## Method 1
    # pair_obj = ak.cartesian([obj_A, obj_B],nested=True)
    # obj1, obj2 = ak.unzip(pair_obj)
    # dr_jm = obj1.delta_r(obj2)
    # min_dr_jm = ak.min(dr_jm,axis=2)
    # mask = min_dr_jm > drmin
    
    ## Method 2
    objB_near, objB_dr = obj_A.nearest(obj_B, return_metric=True)
    mask = ak.fill_none(objB_dr > drmin, True) # I guess to use True is because if there are no objB, all the objA are clean
    return (mask)
    #    dr_photon = selection_cut.DR(gen_photon,pho4.Photon, drmin=0.3)
    #    dr_cut = (ak.num(pho4.Photon[dr_photon]) != 0)