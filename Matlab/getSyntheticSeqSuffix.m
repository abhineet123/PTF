function [ syn_out_suffix ] = getSyntheticSeqSuffix(syn_ssm, syn_ssm_sigma_id,...
    syn_ilm,syn_am_sigma_id, syn_add_noise, syn_noise_mean, syn_noise_sigma)
    syn_out_suffix = sprintf('warped_%s_s%d',syn_ssm, syn_ssm_sigma_id);
    if ~strcmp(syn_ilm, '0')
        syn_out_suffix = sprintf('%s_%s_s%d',syn_out_suffix, syn_ilm, syn_am_sigma_id);
    end
    if syn_add_noise
        syn_out_suffix = sprintf('%s_gauss_%4.2f_%4.2f',...
            syn_out_suffix, syn_noise_mean, syn_noise_sigma);
    end
end

