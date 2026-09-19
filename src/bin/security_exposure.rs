use ccmm_rs::ckks::{
    research_profile_4096, CkksErrorDistribution, CkksSecretDistribution, CkksSecurityExposure,
};

fn main() {
    let profile = research_profile_4096();
    let chain = profile.modulus_chain();

    for level in 0..chain.len() {
        if level > 0 {
            println!();
        }

        let exposure = CkksSecurityExposure::bounded_reference(
            &profile,
            level,
            20,
            CkksSecretDistribution::UniformTernary,
            CkksErrorDistribution::DiscreteGaussian { sigma: 3.19 },
        );

        print!("{}", exposure.to_report_string());
    }
}
