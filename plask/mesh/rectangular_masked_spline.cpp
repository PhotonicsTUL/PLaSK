/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include "rectangular_masked_spline.hpp"
#include "hyman.hpp"
#include "../exceptions.hpp"
#include "../math.hpp"

/*5*/
namespace plask {

#define NOT_INCLUDED CompressedSetOfNumbers<std::size_t>::NOT_INCLUDED


template <typename DstT, typename SrcT>
DstT SplineMaskedRect2DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const
{
    Vec<2> p;
    size_t i0_lo, i0_hi, i1_lo, i1_hi;

    if (!this->src_mesh->prepareInterpolation(this->dst_mesh->at(index), p, i0_lo, i0_hi, i1_lo, i1_hi, this->flags))
        return NaN<decltype(this->src_vec[0])>();

    double left = this->src_mesh->fullMesh.axis[0]->at(i0_lo), right = this->src_mesh->fullMesh.axis[0]->at(i0_hi),
           bottom = this->src_mesh->fullMesh.axis[1]->at(i1_lo), top = this->src_mesh->fullMesh.axis[1]->at(i1_hi);
    double d0 = right - left,
           d1 = top - bottom;
    double x0 = (p.c0 - left) / d0,
           x1 = (p.c1 - bottom) / d1;

    // Hermite 3rd order spline polynomials (in Horner form)
    double hl = ( 2.*x0 - 3.) * x0*x0 + 1.,
           hr = (-2.*x0 + 3.) * x0*x0,
           gl = ((x0 - 2.) * x0 + 1.) * x0 * d0,
           gr = (x0 - 1.) * x0 * x0 * d0,
           hb = ( 2.*x1 - 3.) * x1*x1 + 1.,
           ht = (-2.*x1 + 3.) * x1*x1,
           gb = ((x1 - 2.) * x1 + 1.) * x1 * d1,
           gt = (x1 - 1.) * x1 * x1 * d1;

    std::size_t ilb = this->src_mesh->index(i0_lo, i1_lo),
                ilt = this->src_mesh->index(i0_lo, i1_hi),
                irb = this->src_mesh->index(i0_hi, i1_lo),
                irt = this->src_mesh->index(i0_hi, i1_hi);
    assert(ilb != NOT_INCLUDED);
    assert(ilt != NOT_INCLUDED);
    assert(irb != NOT_INCLUDED);
    assert(irt != NOT_INCLUDED);

    SrcT diff0lb = diff0[ilb],
         diff0lt = diff0[ilt],
         diff0rb = diff0[irb],
         diff0rt = diff0[irt],
         diff1lb = diff1[ilb],
         diff1lt = diff1[ilt],
         diff1rb = diff1[irb],
         diff1rt = diff1[irt];

    SrcT data_lb = this->src_vec[ilb],
         data_lt = this->src_vec[ilt],
         data_rb = this->src_vec[irb],
         data_rt = this->src_vec[irt],
         diff_l = gb * diff1lb + gt * diff1lt,
         diff_r = gb * diff1rb + gt * diff1rt,
         diff_b = gl * diff0lb + gr * diff0rb,
         diff_t = gl * diff0lt + gr * diff0rt;

    return this->flags.postprocess(this->dst_mesh->at(index),
        hl * (hb * data_lb + ht * data_lt) + hr * (hb * data_rb + ht * data_rt) +
        hb * diff_b + ht * diff_t + hl * diff_l + hr * diff_r
    );
}

template <typename DstT, typename SrcT>
DstT SplineMaskedRectElement2DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const
{
    Vec<2> p;
    size_t i0_lo, i0_hi, i1_lo, i1_hi;

    if (!this->src_mesh->prepareInterpolation(this->dst_mesh->at(index), p, i0_lo, i0_hi, i1_lo, i1_hi, this->flags))
        return NaN<decltype(this->src_vec[0])>();

    unsigned char s0, s1; // original index shift

    Vec<2> p_lo = this->src_mesh->fullMesh.at(i0_lo, i1_lo), p_hi;
    if (p.c0 < p_lo.c0) {
        s0 = 1;
        i0_hi = i0_lo;
        if (i0_lo != 0) --i0_lo;
        p_hi.c0 = p_lo.c0;
        p_lo.c0 = this->src_mesh->fullMesh.axis[0]->at(i0_lo);
    } else {
        s0 = 0;
        if (i0_hi == this->src_mesh->fullMesh.axis[0]->size()) --i0_hi;
        p_hi.c0 = this->src_mesh->fullMesh.axis[0]->at(i0_hi);
    }
    if (p.c1 < p_lo.c1) {
        s1 = 2;
        i1_hi = i1_lo;
        if (i1_lo != 0) --i1_lo;
        p_hi.c1 = p_lo.c1;
        p_lo.c1 = this->src_mesh->fullMesh.axis[1]->at(i1_lo);
    } else {
        s1 = 0;
        if (i1_hi == this->src_mesh->fullMesh.axis[1]->size()) --i1_hi;
        p_hi.c1 = this->src_mesh->fullMesh.axis[1]->at(i1_hi);
    }

    // Hermite 3rd order spline polynomials (in Horner form)
    double d0 = p_hi.c0 - p_lo.c0,
           d1 = p_hi.c1 - p_lo.c1;
    double x0 = (i0_lo != i0_hi)? (p.c0 - p_lo.c0) / d0 : 0./*5*/,
           x1 = (i1_lo != i1_hi)? (p.c1 - p_lo.c1) / d1 : 0./*5*/;

    // Hermite 3rd order spline polynomials (in Horner form)
    double hl = ( 2.*x0 - 3.) * x0*x0 + 1.,
           hr = (-2.*x0 + 3.) * x0*x0,
           gl = ((x0 - 2.) * x0 + 1.) * x0 * d0,
           gr = (x0 - 1.) * x0 * x0 * d0,
           hb = ( 2.*x1 - 3.) * x1*x1 + 1.,
           ht = (-2.*x1 + 3.) * x1*x1,
           gb = ((x1 - 2.) * x1 + 1.) * x1 * d1,
           gt = (x1 - 1.) * x1 * x1 * d1;

    size_t idx[] = {this->src_mesh->index(i0_lo, i1_lo),    // lb
                    this->src_mesh->index(i0_hi, i1_lo),    // rb
                    this->src_mesh->index(i0_lo, i1_hi),    // lt
                    this->src_mesh->index(i0_hi, i1_hi)};   // rt

    SrcT diff0lb = (idx[0] != NOT_INCLUDED)? diff0[idx[0]] : Zero<SrcT>(),
         diff0rb = (idx[1] != NOT_INCLUDED)? diff0[idx[1]] : Zero<SrcT>(),
         diff0lt = (idx[2] != NOT_INCLUDED)? diff0[idx[2]] : Zero<SrcT>(),
         diff0rt = (idx[3] != NOT_INCLUDED)? diff0[idx[3]] : Zero<SrcT>(),
         diff1lb = (idx[0] != NOT_INCLUDED)? diff1[idx[0]] : Zero<SrcT>(),
         diff1rb = (idx[1] != NOT_INCLUDED)? diff1[idx[1]] : Zero<SrcT>(),
         diff1lt = (idx[2] != NOT_INCLUDED)? diff1[idx[2]] : Zero<SrcT>(),
         diff1rt = (idx[3] != NOT_INCLUDED)? diff1[idx[3]] : Zero<SrcT>();

    typename std::remove_const<SrcT>::type vals[4];

    size_t iaa = idx[0+s0+s1],
           iba = idx[1-s0+s1],
           iab = idx[2+s0-s1],
           ibb = idx[3-s0-s1];
    typename std::remove_const<SrcT>::type
           &val_aa = vals[0+s0+s1],
           &val_ba = vals[1-s0+s1],
           &val_ab = vals[2+s0-s1],
           &val_bb = vals[3-s0-s1];

    val_aa = this->src_vec[iaa];
    val_ab = (iab != NOT_INCLUDED)? this->src_vec[iab] : val_aa;
    val_ba = (iba != NOT_INCLUDED)? this->src_vec[iba] : val_aa;
    val_bb = (ibb != NOT_INCLUDED)? this->src_vec[ibb] : 0.5 * (val_ab + val_ba);


    SrcT diff_l = gb * diff1lb + gt * diff1lt,
         diff_r = gb * diff1rb + gt * diff1rt,
         diff_b = gl * diff0lb + gr * diff0rb,
         diff_t = gl * diff0lt + gr * diff0rt;

    SrcT result = hl * (hb * vals[0] + ht * vals[2]) + hr * (hb * vals[1] + ht * vals[3]) +
        hb * diff_b + ht * diff_t + hl * diff_l + hr * diff_r;

    return this->flags.postprocess(this->dst_mesh->at(index), result);
}


template <typename DstT, typename SrcT>
DstT SplineMaskedRect3DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const
{
    Vec<3> p;
    size_t i0_lo, i0_hi, i1_lo, i1_hi, i2_lo, i2_hi;

    if (!this->src_mesh->prepareInterpolation(this->dst_mesh->at(index), p, i0_lo, i0_hi, i1_lo, i1_hi, i2_lo, i2_hi, this->flags))
        return NaN<decltype(this->src_vec[0])>();

    double back = this->src_mesh->fullMesh.axis[0]->at(i0_lo), front = this->src_mesh->fullMesh.axis[0]->at(i0_hi),
           left = this->src_mesh->fullMesh.axis[1]->at(i1_lo), right = this->src_mesh->fullMesh.axis[1]->at(i1_hi),
           bottom = this->src_mesh->fullMesh.axis[2]->at(i2_lo), top = this->src_mesh->fullMesh.axis[2]->at(i2_hi);

    std::size_t illl = this->src_mesh->index(i0_lo, i1_lo, i2_lo),
                illh = this->src_mesh->index(i0_lo, i1_lo, i2_hi),
                ilhl = this->src_mesh->index(i0_lo, i1_hi, i2_lo),
                ilhh = this->src_mesh->index(i0_lo, i1_hi, i2_hi),
                ihll = this->src_mesh->index(i0_hi, i1_lo, i2_lo),
                ihlh = this->src_mesh->index(i0_hi, i1_lo, i2_hi),
                ihhl = this->src_mesh->index(i0_hi, i1_hi, i2_lo),
                ihhh = this->src_mesh->index(i0_hi, i1_hi, i2_hi);
    assert(illl != NOT_INCLUDED);
    assert(illh != NOT_INCLUDED);
    assert(ilhl != NOT_INCLUDED);
    assert(ilhh != NOT_INCLUDED);
    assert(ihll != NOT_INCLUDED);
    assert(ihlh != NOT_INCLUDED);
    assert(ihhl != NOT_INCLUDED);
    assert(ihhh != NOT_INCLUDED);

    double d0 = front - back,
           d1 = right - left,
           d2 = top - bottom;
    double x0 = (p.c0 - back) / d0,
           x1 = (p.c1 - left) / d1,
           x2 = (p.c2 - bottom) / d2;

    // Hermite 3rd order spline polynomials (in Horner form)
    double h0l = ( 2.*x0 - 3.) * x0*x0 + 1.,
           h0h = (-2.*x0 + 3.) * x0*x0,
           g0l = ((x0 - 2.) * x0 + 1.) * x0 * d0,
           g0h = (x0 - 1.) * x0 * x0 * d0,
           h1l = ( 2.*x1 - 3.) * x1*x1 + 1.,
           h1h = (-2.*x1 + 3.) * x1*x1,
           g1l = ((x1 - 2.) * x1 + 1.) * x1 * d1,
           g1h = (x1 - 1.) * x1 * x1 * d1,
           h2l = ( 2.*x2 - 3.) * x2*x2 + 1.,
           h2h = (-2.*x2 + 3.) * x2*x2,
           g2l = ((x2 - 2.) * x2 + 1.) * x2 * d2,
           g2h = (x2 - 1.) * x2 * x2 * d2;

    SrcT diff0lll = diff0[illl],
         diff0llh = diff0[illh],
         diff0lhl = diff0[ilhl],
         diff0lhh = diff0[ilhh],
         diff0hll = diff0[ihll],
         diff0hlh = diff0[ihlh],
         diff0hhl = diff0[ihhl],
         diff0hhh = diff0[ihhh],
         diff1lll = diff1[illl],
         diff1llh = diff1[illh],
         diff1lhl = diff1[ilhl],
         diff1lhh = diff1[ilhh],
         diff1hll = diff1[ihll],
         diff1hlh = diff1[ihlh],
         diff1hhl = diff1[ihhl],
         diff1hhh = diff1[ihhh],
         diff2lll = diff2[illl],
         diff2llh = diff2[illh],
         diff2lhl = diff2[ilhl],
         diff2lhh = diff2[ilhh],
         diff2hll = diff2[ihll],
         diff2hlh = diff2[ihlh],
         diff2hhl = diff2[ihhl],
         diff2hhh = diff2[ihhh];

    SrcT data_lll = this->src_vec[illl],
         data_llh = this->src_vec[illh],
         data_lhl = this->src_vec[ilhl],
         data_lhh = this->src_vec[ilhh],
         data_hll = this->src_vec[ihll],
         data_hlh = this->src_vec[ihlh],
         data_hhl = this->src_vec[ihhl],
         data_hhh = this->src_vec[ihhh],
         D_ll = g0l * diff0lll + g0h * diff0hll, Dl_l = g1l * diff1lll + g1h * diff1lhl, Dll_ = g2l * diff2lll + g2h * diff2llh,
         D_lh = g0l * diff0llh + g0h * diff0hlh, Dl_h = g1l * diff1llh + g1h * diff1lhh, Dlh_ = g2l * diff2lhl + g2h * diff2lhh,
         D_hl = g0l * diff0lhl + g0h * diff0hhl, Dh_l = g1l * diff1hll + g1h * diff1hhl, Dhl_ = g2l * diff2hll + g2h * diff2hlh,
         D_hh = g0l * diff0lhh + g0h * diff0hhh, Dh_h = g1l * diff1hlh + g1h * diff1hhh, Dhh_ = g2l * diff2hhl + g2h * diff2hhh;

    return this->flags.postprocess(this->dst_mesh->at(index),
        h0l * h1l * h2l * data_lll +
        h0l * h1l * h2h * data_llh +
        h0l * h1h * h2l * data_lhl +
        h0l * h1h * h2h * data_lhh +
        h0h * h1l * h2l * data_hll +
        h0h * h1l * h2h * data_hlh +
        h0h * h1h * h2l * data_hhl +
        h0h * h1h * h2h * data_hhh +
        h1l * h2l * D_ll + h0l * h2l * Dl_l + h0l * h1l * Dll_ +
        h1l * h2h * D_lh + h0l * h2h * Dl_h + h0l * h1h * Dlh_ +
        h1h * h2l * D_hl + h0h * h2l * Dh_l + h0h * h1l * Dhl_ +
        h1h * h2h * D_hh + h0h * h2h * Dh_h + h0h * h1h * Dhh_
    );
}

template <typename DstT, typename SrcT>
DstT SplineMaskedRectElement3DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const
{
    Vec<3> p;
    size_t i0_lo, i0_hi, i1_lo, i1_hi, i2_lo, i2_hi;

    if (!this->src_mesh->prepareInterpolation(this->dst_mesh->at(index), p, i0_lo, i0_hi, i1_lo, i1_hi, i2_lo, i2_hi, this->flags))
        return NaN<decltype(this->src_vec[0])>();

    unsigned char s0, s1, s2; // original index shift

    Vec<3> p_lo = this->src_mesh->fullMesh.at(i0_lo, i1_lo, i2_lo), p_hi;
    if (p.c0 < p_lo.c0) {
        s0 = 4;
        i0_hi = i0_lo;
        if (i0_lo != 0) --i0_lo;
        p_hi.c0 = p_lo.c0;
        p_lo.c0 = this->src_mesh->fullMesh.axis[0]->at(i0_lo);
    } else {
        s0 = 0;
        if (i0_hi == this->src_mesh->fullMesh.axis[0]->size()) --i0_hi;
        p_hi.c0 = this->src_mesh->fullMesh.axis[0]->at(i0_hi);
    }
    if (p.c1 < p_lo.c1) {
        s1 = 2;
        i1_hi = i1_lo;
        if (i1_lo != 0) --i1_lo;
        p_hi.c1 = p_lo.c1;
        p_lo.c1 = this->src_mesh->fullMesh.axis[1]->at(i1_lo);
    } else {
        s1 = 0;
        if (i1_hi == this->src_mesh->fullMesh.axis[1]->size()) --i1_hi;
        p_hi.c1 = this->src_mesh->fullMesh.axis[1]->at(i1_hi);
    }
    if (p.c2 < p_lo.c2) {
        s2 = 1;
        i2_hi = i2_lo;
        if (i2_lo != 0) --i2_lo;
        p_hi.c2 = p_lo.c2;
        p_lo.c2 = this->src_mesh->fullMesh.axis[2]->at(i2_lo);
    } else {
        s2 = 0;
        if (i2_hi == this->src_mesh->fullMesh.axis[2]->size()) --i2_hi;
        p_hi.c2 = this->src_mesh->fullMesh.axis[2]->at(i2_hi);
    }

    double d0 = p_hi.c0 - p_lo.c0,
           d1 = p_hi.c1 - p_lo.c1,
           d2 = p_hi.c2 - p_lo.c2;
    double x0 = (i0_lo != i0_hi)? (p.c0 - p_lo.c0) / d0 : 0./*5*/,
           x1 = (i1_lo != i1_hi)? (p.c1 - p_lo.c1) / d1 : 0./*5*/,
           x2 = (i2_lo != i2_hi)? (p.c2 - p_lo.c2) / d2 : 0./*5*/;

    // Hermite 3rd order spline polynomials (in Horner form)
    double h0l = ( 2.*x0 - 3.) * x0*x0 + 1.,
           h0h = (-2.*x0 + 3.) * x0*x0,
           g0l = ((x0 - 2.) * x0 + 1.) * x0 * d0,
           g0h = (x0 - 1.) * x0 * x0 * d0,
           h1l = ( 2.*x1 - 3.) * x1*x1 + 1.,
           h1h = (-2.*x1 + 3.) * x1*x1,
           g1l = ((x1 - 2.) * x1 + 1.) * x1 * d1,
           g1h = (x1 - 1.) * x1 * x1 * d1,
           h2l = ( 2.*x2 - 3.) * x2*x2 + 1.,
           h2h = (-2.*x2 + 3.) * x2*x2,
           g2l = ((x2 - 2.) * x2 + 1.) * x2 * d2,
           g2h = (x2 - 1.) * x2 * x2 * d2;

    std::size_t idx[] = {this->src_mesh->index(i0_lo, i1_lo, i2_lo),
                         this->src_mesh->index(i0_lo, i1_lo, i2_hi),
                         this->src_mesh->index(i0_lo, i1_hi, i2_lo),
                         this->src_mesh->index(i0_lo, i1_hi, i2_hi),
                         this->src_mesh->index(i0_hi, i1_lo, i2_lo),
                         this->src_mesh->index(i0_hi, i1_lo, i2_hi),
                         this->src_mesh->index(i0_hi, i1_hi, i2_lo),
                         this->src_mesh->index(i0_hi, i1_hi, i2_hi)};

    SrcT diff0lll = (idx[0] != NOT_INCLUDED)? diff0[idx[0]/*lll*/] : Zero<SrcT>(),
         diff0llh = (idx[1] != NOT_INCLUDED)? diff0[idx[1]/*llh*/] : Zero<SrcT>(),
         diff0lhl = (idx[2] != NOT_INCLUDED)? diff0[idx[2]/*lhl*/] : Zero<SrcT>(),
         diff0lhh = (idx[3] != NOT_INCLUDED)? diff0[idx[3]/*lhh*/] : Zero<SrcT>(),
         diff0hll = (idx[4] != NOT_INCLUDED)? diff0[idx[4]/*hll*/] : Zero<SrcT>(),
         diff0hlh = (idx[5] != NOT_INCLUDED)? diff0[idx[5]/*hlh*/] : Zero<SrcT>(),
         diff0hhl = (idx[6] != NOT_INCLUDED)? diff0[idx[6]/*hhl*/] : Zero<SrcT>(),
         diff0hhh = (idx[7] != NOT_INCLUDED)? diff0[idx[7]/*hhh*/] : Zero<SrcT>(),
         diff1lll = (idx[0] != NOT_INCLUDED)? diff1[idx[0]/*lll*/] : Zero<SrcT>(),
         diff1llh = (idx[1] != NOT_INCLUDED)? diff1[idx[1]/*llh*/] : Zero<SrcT>(),
         diff1lhl = (idx[2] != NOT_INCLUDED)? diff1[idx[2]/*lhl*/] : Zero<SrcT>(),
         diff1lhh = (idx[3] != NOT_INCLUDED)? diff1[idx[3]/*lhh*/] : Zero<SrcT>(),
         diff1hll = (idx[4] != NOT_INCLUDED)? diff1[idx[4]/*hll*/] : Zero<SrcT>(),
         diff1hlh = (idx[5] != NOT_INCLUDED)? diff1[idx[5]/*hlh*/] : Zero<SrcT>(),
         diff1hhl = (idx[6] != NOT_INCLUDED)? diff1[idx[6]/*hhl*/] : Zero<SrcT>(),
         diff1hhh = (idx[7] != NOT_INCLUDED)? diff1[idx[7]/*hhh*/] : Zero<SrcT>(),
         diff2lll = (idx[0] != NOT_INCLUDED)? diff2[idx[0]/*lll*/] : Zero<SrcT>(),
         diff2llh = (idx[1] != NOT_INCLUDED)? diff2[idx[1]/*llh*/] : Zero<SrcT>(),
         diff2lhl = (idx[2] != NOT_INCLUDED)? diff2[idx[2]/*lhl*/] : Zero<SrcT>(),
         diff2lhh = (idx[3] != NOT_INCLUDED)? diff2[idx[3]/*lhh*/] : Zero<SrcT>(),
         diff2hll = (idx[4] != NOT_INCLUDED)? diff2[idx[4]/*hll*/] : Zero<SrcT>(),
         diff2hlh = (idx[5] != NOT_INCLUDED)? diff2[idx[5]/*hlh*/] : Zero<SrcT>(),
         diff2hhl = (idx[6] != NOT_INCLUDED)? diff2[idx[6]/*hhl*/] : Zero<SrcT>(),
         diff2hhh = (idx[7] != NOT_INCLUDED)? diff2[idx[7]/*hhh*/] : Zero<SrcT>();

    typename std::remove_const<SrcT>::type vals[8];

    size_t iaaa = idx[0+s0+s1+s2],
           iaab = idx[1+s0+s1-s2],
           iaba = idx[2+s0-s1+s2],
           iabb = idx[3+s0-s1-s2],
           ibaa = idx[4-s0+s1+s2],
           ibab = idx[5-s0+s1-s2],
           ibba = idx[6-s0-s1+s2],
           ibbb = idx[7-s0-s1-s2];
    typename std::remove_const<SrcT>::type
           &val_aaa = vals[0+s0+s1+s2],
           &val_aab = vals[1+s0+s1-s2],
           &val_aba = vals[2+s0-s1+s2],
           &val_abb = vals[3+s0-s1-s2],
           &val_baa = vals[4-s0+s1+s2],
           &val_bab = vals[5-s0+s1-s2],
           &val_bba = vals[6-s0-s1+s2],
           &val_bbb = vals[7-s0-s1-s2];

    val_aaa = this->src_vec[iaaa];
    val_aab = (iaab != NOT_INCLUDED)? this->src_vec[iaab] : val_aaa;
    val_aba = (iaba != NOT_INCLUDED)? this->src_vec[iaba] : val_aaa;
    val_baa = (ibaa != NOT_INCLUDED)? this->src_vec[ibaa] : val_aaa;
    val_abb = (iabb != NOT_INCLUDED)? this->src_vec[iabb] : 0.5 * (val_aab + val_aba);
    val_bab = (ibab != NOT_INCLUDED)? this->src_vec[ibab] : 0.5 * (val_aab + val_baa);
    val_bba = (ibba != NOT_INCLUDED)? this->src_vec[ibba] : 0.5 * (val_aba + val_baa);
    val_bbb = (ibbb != NOT_INCLUDED)? this->src_vec[ibbb] : (val_abb + val_bab + val_bba) / 3.;

    SrcT D_ll = g0l * diff0lll + g0h * diff0hll, Dl_l = g1l * diff1lll + g1h * diff1lhl, Dll_ = g2l * diff2lll + g2h * diff2llh,
         D_lh = g0l * diff0llh + g0h * diff0hlh, Dl_h = g1l * diff1llh + g1h * diff1lhh, Dlh_ = g2l * diff2lhl + g2h * diff2lhh,
         D_hl = g0l * diff0lhl + g0h * diff0hhl, Dh_l = g1l * diff1hll + g1h * diff1hhl, Dhl_ = g2l * diff2hll + g2h * diff2hlh,
         D_hh = g0l * diff0lhh + g0h * diff0hhh, Dh_h = g1l * diff1hlh + g1h * diff1hhh, Dhh_ = g2l * diff2hhl + g2h * diff2hhh;

    return this->flags.postprocess(this->dst_mesh->at(index),
        h0l * h1l * h2l * vals[0/*lll*/] +
        h0l * h1l * h2h * vals[1/*llh*/] +
        h0l * h1h * h2l * vals[2/*lhl*/] +
        h0l * h1h * h2h * vals[3/*lhh*/] +
        h0h * h1l * h2l * vals[4/*hll*/] +
        h0h * h1l * h2h * vals[5/*hlh*/] +
        h0h * h1h * h2l * vals[6/*hhl*/] +
        h0h * h1h * h2h * vals[7/*hhh*/] +
        h1l * h2l * D_ll + h0l * h2l * Dl_l + h0l * h1l * Dll_ +
        h1l * h2h * D_lh + h0l * h2h * Dl_h + h0l * h1h * Dlh_ +
        h1h * h2l * D_hl + h0h * h2l * Dh_l + h0h * h1l * Dhl_ +
        h1h * h2h * D_hh + h0h * h2h * Dh_h + h0h * h1h * Dhh_
    );
}


namespace masked_hyman {
    template <typename DataT, typename IndexF>
    static void computeDiffs(DataT* diffs, int ax, const shared_ptr<MeshAxis>& axis,
                             const DataT* data, IndexF&& index, const InterpolationFlags& flags)
    {
        const std::size_t n1 = axis->size() - 1;

        for (std::size_t i = 1; i != n1; ++i) {
            const std::size_t idx = index(i);
            if (idx == NOT_INCLUDED) continue;
            const double da = axis->at(i) - axis->at(i-1),
                         db = axis->at(i+1) - axis->at(i);
            size_t idxm = index(i-1), idxp = index(i+1);
            if (idxm == NOT_INCLUDED || idxp == NOT_INCLUDED) {   // at edges derivative is 0
                diffs[idx] = Zero<DataT>();
            } else {
                const DataT sa = (data[idx] - data[idxm]) / da,
                            sb = (data[idxp] - data[idx]) / db;
                // Use parabolic estimation of the derivative
                diffs[idx] = (da * sb  + db * sa) / (da + db);
                // Hyman filter
                Hyman<DataT>::filter(diffs[idx], sa, sb);
            }
        }

        const size_t i0 = index(0), i1 = index(1), in = index(n1), in1 = index(n1-1);
        double da0, db0, dan, dbn;
        DataT sa0, sb0, san, sbn;

        if (i0 != NOT_INCLUDED) {
            if (flags.symmetric(ax) && i1 != NOT_INCLUDED) {
                da0 = axis->at(0);
                db0 = axis->at(1) - axis->at(0);
                sb0 = (data[i1] - data[i0]) / db0;
                if (da0 < 0. && flags.periodic(ax)) {
                    da0 += flags.high(ax) - flags.low(ax);
                }
                if (da0 == 0.)
                    sa0 = (data[i1] - flags.reflect(ax, data[i1])) / (2.*db0);
                else if (da0 > 0.)
                    sa0 = (data[i0] - flags.reflect(ax, data[i0])) / (2.*da0);
                else {
                    da0 = db0 = 0.5;
                    sa0 = sb0 = Zero<DataT>();
                }
            } else {
                da0 = db0 = 0.5;
                sa0 = sb0 = Zero<DataT>();
            }

            // Use parabolic estimation of the derivative with Hyman filter
            diffs[i0] = (da0 * sb0 + db0 * sa0) / (da0 + db0);
            Hyman<DataT>::filter(diffs[i0], sa0, sb0);
        }

        if (in != NOT_INCLUDED) {
            if (flags.symmetric(ax) && in1 != NOT_INCLUDED) {
                dan = axis->at(n1) - axis->at(n1-1);
                san = (data[in] - data[in1]) / dan;
                dbn = - axis->at(n1);
                if (dbn < 0. && flags.periodic(ax)) {
                    dbn += flags.high(ax) - flags.low(ax);
                }
                if (dbn == 0.)
                    sbn = (data[in1] - flags.reflect(ax, data[in1])) / (2.*dan);
                else if (dbn > 0.)
                    sbn = (data[in] - flags.reflect(ax, data[in])) / (2.*dbn);
                else {
                    dan = dbn = 0.5;
                    san = sbn = Zero<DataT>();
                }
            } else {
                dan = dbn = 0.5;
                san = sbn = Zero<DataT>();
            }

            // Use parabolic estimation of the derivative with Hyman filter
            diffs[in] = (dan * sbn + dbn * san) / (dan + dbn);
            Hyman<DataT>::filter(diffs[in], san, sbn);
        }
    }
}


template <typename DstT, typename SrcT, typename BaseT>
HymanSplineMaskedRect2DLazyDataImpl<DstT, SrcT, BaseT>::HymanSplineMaskedRect2DLazyDataImpl(const shared_ptr<const typename BaseT::MeshType>& src_mesh,
                                                                                            const DataVector<const SrcT>& src_vec,
                                                                                            const shared_ptr<const MeshD<2>>& dst_mesh,
                                                                                            const InterpolationFlags& flags):
    BaseT(src_mesh, src_vec, dst_mesh, flags) {
    const size_t n0 = src_mesh->fullMesh.axis[0]->size(), n1 = src_mesh->fullMesh.axis[1]->size();

    if (n0 == 0 || n1 == 0)
        throw BadMesh("interpolate", "source mesh empty");

    if (n0 > 1)
        for (size_t i1 = 0; i1 < n1; ++i1)
            masked_hyman::computeDiffs<SrcT>(this->diff0.data(), 0, src_mesh->fullMesh.axis[0], src_vec.data(),
                                             [&src_mesh, i1](size_t i0) -> size_t { return src_mesh->index(i0, i1); },
                                             flags);
    else
        std::fill(this->diff0.begin(), this->diff0.end(), Zero<SrcT>());
    if (n1 > 1)
        for (size_t i0 = 0; i0 < n0; ++i0)
            masked_hyman::computeDiffs<SrcT>(this->diff1.data(), 1, src_mesh->fullMesh.axis[1], src_vec.data(),
                                             [&src_mesh, i0](size_t i1) -> size_t { return src_mesh->index(i0, i1); },
                                             flags);
    else
        std::fill(this->diff1.begin(), this->diff1.end(), Zero<SrcT>());
}



template <typename DstT, typename SrcT, typename BaseT>
HymanSplineMaskedRect3DLazyDataImpl<DstT, SrcT, BaseT>::HymanSplineMaskedRect3DLazyDataImpl(const shared_ptr<const typename BaseT::MeshType>& src_mesh,
                                                                                            const DataVector<const SrcT>& src_vec,
                                                                                            const shared_ptr<const MeshD<3>>& dst_mesh,
                                                                                            const InterpolationFlags& flags):
    BaseT(src_mesh, src_vec, dst_mesh, flags) {
    const size_t n0 = src_mesh->fullMesh.axis[0]->size(), n1 = src_mesh->fullMesh.axis[1]->size(), n2 = src_mesh->fullMesh.axis[2]->size();

    if (n0 == 0 || n1 == 0 || n2 == 0)
        throw BadMesh("interpolate", "source mesh empty");

    if (n0 > 1) {
        for (size_t i2 = 0; i2 < n2; ++i2) {
            for (size_t i1 = 0; i1 < n1; ++i1) {
                masked_hyman::computeDiffs<SrcT>(this->diff0.data(), 0, src_mesh->fullMesh.axis[0], src_vec.data(),
                                                 [&src_mesh, i2, i1](size_t i0) -> size_t { return src_mesh->index(i0, i1, i2); },
                                                 flags);
            }
        }
    } else
        std::fill(this->diff0.begin(), this->diff0.end(), Zero<SrcT>());

    if (n1 > 1) {
        for (size_t i2 = 0; i2 < n2; ++i2) {
            for (size_t i0 = 0; i0 < n0; ++i0) {
                masked_hyman::computeDiffs<SrcT>(this->diff1.data(), 1, src_mesh->fullMesh.axis[1], src_vec.data(),
                                               [&src_mesh, i2, i0](size_t i1) -> size_t { return src_mesh->index(i0, i1, i2); },
                                               flags);
            }
        }
    } else
        std::fill(this->diff1.begin(), this->diff1.end(), Zero<SrcT>());

    if (n2 > 1) {
        for (size_t i1 = 0; i1 < n1; ++i1) {
            for (size_t i0 = 0; i0 < n0; ++i0) {
                masked_hyman::computeDiffs<SrcT>(this->diff2.data(), 2, src_mesh->fullMesh.axis[2], src_vec.data(),
                                                 [&src_mesh, i1, i0](size_t i2) -> size_t { return src_mesh->index(i0, i1, i2); },
                                                 flags);
            }
        }
    } else
        std::fill(this->diff2.begin(), this->diff2.end(), Zero<SrcT>());

}


template struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl<double, double>;
template struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl<dcomplex, dcomplex>;

template struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl<Vec<2,double>, Vec<2,double>>;
template struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>>;

template struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl<Vec<3,double>, Vec<3,double>>;
template struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>>;

template struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl<Tensor2<double>, Tensor2<double>>;
template struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>>;

template struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl<Tensor3<double>, Tensor3<double>>;
template struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>>;

template struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl<double, double>;
template struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl<dcomplex, dcomplex>;

template struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl<Vec<2,double>, Vec<2,double>>;
template struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>>;

template struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl<Vec<3,double>, Vec<3,double>>;
template struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>>;

template struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl<Tensor2<double>, Tensor2<double>>;
template struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>>;

template struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl<Tensor3<double>, Tensor3<double>>;
template struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>>;


template struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl<double, double, SplineMaskedRectElement2DLazyDataImpl<double, double>>;
template struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl<dcomplex, dcomplex, SplineMaskedRectElement2DLazyDataImpl<dcomplex, dcomplex>>;

template struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl<Vec<2,double>, Vec<2,double>, SplineMaskedRectElement2DLazyDataImpl<Vec<2,double>, Vec<2,double>>>;
template struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>, SplineMaskedRectElement2DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>>>;

template struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl<Vec<3,double>, Vec<3,double>, SplineMaskedRectElement2DLazyDataImpl<Vec<3,double>, Vec<3,double>>>;
template struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>, SplineMaskedRectElement2DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>>>;

template struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl<Tensor2<double>, Tensor2<double>, SplineMaskedRectElement2DLazyDataImpl<Tensor2<double>, Tensor2<double>>>;
template struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>, SplineMaskedRectElement2DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>>>;

template struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl<Tensor3<double>, Tensor3<double>, SplineMaskedRectElement2DLazyDataImpl<Tensor3<double>, Tensor3<double>>>;
template struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>, SplineMaskedRectElement2DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>>>;

template struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl<double, double, SplineMaskedRectElement3DLazyDataImpl<double, double>>;
template struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl<dcomplex, dcomplex, SplineMaskedRectElement3DLazyDataImpl<dcomplex, dcomplex>>;

template struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl<Vec<2,double>, Vec<2,double>, SplineMaskedRectElement3DLazyDataImpl<Vec<2,double>, Vec<2,double>>>;
template struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>, SplineMaskedRectElement3DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>>>;

template struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl<Vec<3,double>, Vec<3,double>, SplineMaskedRectElement3DLazyDataImpl<Vec<3,double>, Vec<3,double>>>;
template struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>, SplineMaskedRectElement3DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>>>;

template struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl<Tensor2<double>, Tensor2<double>, SplineMaskedRectElement3DLazyDataImpl<Tensor2<double>, Tensor2<double>>>;
template struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>, SplineMaskedRectElement3DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>>>;

template struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl<Tensor3<double>, Tensor3<double>, SplineMaskedRectElement3DLazyDataImpl<Tensor3<double>, Tensor3<double>>>;
template struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>, SplineMaskedRectElement3DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>>>;




} // namespace plask
