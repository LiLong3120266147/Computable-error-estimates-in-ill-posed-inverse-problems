% 4.2(2)：统一真解约束下的有限维最优恢复实验
%
% 将 n = 32,64,96,128,160,192,224,256 解释为“保留的 Fourier 模态总数”
% 固定 256x256 为参考原问题，因此参考谱空间共有 256^2=65536 个 Fourier 模态。
% 对所有有限维问题统一使用：
%   R = Omega(z_true)                         （同一个真解先验半径）
%   Delta(delta) = delta*||u_true||           （同一个信息误差半径）
%   ell = 256x256 参考网格上的 D1+D2 区域平均泛函
% 最终采用相对最优恢复误差 e_n^opt = E_n^opt / |ell(z_true)|
% 进行展示，便于不同噪声水平下比较
%
% 在“统一约束 + 嵌套有限维子空间”的设置下, E_n^opt <= E_ref^opt,
% 且随着保留模态数 n 增加，E_n^opt 从下方趋近 256x256 参考问题的 E_ref^opt。
% 因而真正随 n 减小的是有限维逼近误差 gap_n = E_ref^opt - E_n^opt.

clc; clear; close all;

%% ========================================================================
%  0. 参数
% =========================================================================
par.zeta = 0.15;
par.nu = 0.25;
par.seed = 1218;

par.N_fine = 512;
par.N_ref = 256;

% 这里是 Fourier 模态总数，不是 n x n 网格
par.n_mode_list = [32 64 96 128 160 192 224 256];

par.delta_list = [0.01 0.03 0.05 0.07];

par.region1 = [0.20,0.44,0.20,0.44];
par.region2 = [0.45,0.75,0.45,0.75];

par.opt_logt_min = -40;
par.opt_logt_max = 40;
par.opt_grid_size = 241;
par.opt_tol_x = 1.0e-10;

L = 1.0;
Nf = par.N_fine;
Nref = par.N_ref;

%% ========================================================================
%  1. 构造 512x512 周期 Fourier 参考场
% =========================================================================
xf = (0:Nf-1)/Nf;
[Xf,Yf] = meshgrid(xf,xf);

T1 = (Xf-0.32).^2+(Yf-0.32).^2 < 0.0004;
T2 = (Xf-0.60).^2-(Xf-0.60).*(Yf-0.60)+(Yf-0.60).^2 < 0.01;

w_fine = 20*double(T1)+double(T2);

[k2_fine,kabs_fine] = fourier_frequencies_2d(Nf,L);

what_fine = fft2(w_fine);

ztrue_fine_hat = exp(-par.zeta*kabs_fine).*what_fine;
z_true_fine = real(ifft2(ztrue_fine_hat));

Ahat_fine = exp(-(par.nu-par.zeta)*kabs_fine);
u_true_fine = real(ifft2(Ahat_fine.*ztrue_fine_hat));

%% ========================================================================
%  2. 固定 256x256 参考原问题
% =========================================================================
if mod(Nf,Nref)~=0
    error('N_fine 必须能被 N_ref 整除。');
end

stride = Nf/Nref;

z_true = z_true_fine(1:stride:end,1:stride:end);
u_true = u_true_fine(1:stride:end,1:stride:end);

ztrue_hat = fft2(z_true);

[k2,kabs] = fourier_frequencies_2d(Nref,L);

Ahat = exp(-(par.nu-par.zeta)*kabs);
Rhat = 1+k2.^2;

% 所有有限维问题统一使用同一个真解先验半径
R_prior = spectral_omega(ztrue_hat,Rhat,Nref);

% 所有有限维问题统一使用同一个真数据范数
u_true_norm = grid_norm(u_true);

fprintf('============================================================\n');
fprintf('256x256 参考原问题\n');
fprintf('============================================================\n');
fprintf('R = Omega(z_true)       = %.12e\n',R_prior);
fprintf('||u_true||              = %.12e\n',u_true_norm);
fprintf('参考 Fourier 模态总数  = %d\n',Nref^2);

%% ========================================================================
%  3. 固定 256x256 参考问题上的 D1+D2 区域平均泛函
% =========================================================================
x = (0:Nref-1)/Nref;
[X,Y] = meshgrid(x,x);

b1 = par.region1;
b2 = par.region2;

mask1 = X>=b1(1) & X<=b1(2) & Y>=b1(3) & Y<=b1(4);
mask2 = X>=b2(1) & X<=b2(2) & Y>=b2(3) & Y<=b2(4);
mask_region = mask1 | mask2;

weights = double(mask_region);
weights = weights/sum(weights(:));

chat = fft2(weights);

functional_true = sum(weights(:).*z_true(:));

fprintf('ell(z_true)             = %.12e\n',functional_true);
fprintf('============================================================\n\n');

%% ========================================================================
%  4. 构造嵌套的低频 Fourier 模态序列
% =========================================================================
% 按 |omega| 从小到大排列参考 256x256 Fourier 模态
% 对 n_mode_list 中每一个 n，只保留前 n 个最低频模态
% 因而有
%   V_32 subset V_64 subset ... subset V_256 subset V_ref.
% 这里采用 Fourier 系数空间进行最优恢复误差计算

[~,mode_order] = sort(kabs(:),'ascend');

n_mode = numel(par.n_mode_list);
n_delta = numel(par.delta_list);

Eopt = zeros(n_mode,n_delta);

% 保存每个有限维子空间中最高保留频率，便于诊断
omega_max_used = zeros(n_mode,1);

%% ========================================================================
%  5. 保留随机种子设置
% =========================================================================
% 当前最优恢复误差只使用误差球半径  Delta = delta*||u_true||，
% 随机噪声方向本身不进入 Eopt 的计算
rng(par.seed,'twister');

%% ========================================================================
%  6. 统一约束下的有限维最优恢复误差
% =========================================================================
for in = 1:n_mode

    n = par.n_mode_list(in);

    if n>Nref^2
        error('保留模态数 n=%d 超过参考问题模态总数 %d。',n,Nref^2);
    end

    spectral_mask = false(Nref,Nref);
    spectral_mask(mode_order(1:n)) = true;

    omega_max_used(in) = max(kabs(spectral_mask));

    if nnz(spectral_mask)~=n
        error('n=%d 时实际保留的 Fourier 模态数不等于 n。',n);
    end

    for id = 1:n_delta

        delta = par.delta_list(id);

        % 所有有限维问题统一采用 256x256 真数据给出的误差半径
        Delta_info = delta*u_true_norm;

        Eopt(in,id) = optimal_recovery_error_masked( ...
            Ahat,Rhat,chat,spectral_mask, ...
            R_prior,Delta_info,par);
    end
end

%% ========================================================================
%  7. 计算完整 256x256 参考问题的最优恢复误差
% =========================================================================
full_mask = true(Nref,Nref);

Eref = zeros(1,n_delta);

for id = 1:n_delta

    delta = par.delta_list(id);
    Delta_info = delta*u_true_norm;

    Eref(id) = optimal_recovery_error_masked( ...
        Ahat,Rhat,chat,full_mask,R_prior,Delta_info,par);
end

% 有限维逼近误差
gap = repmat(Eref,n_mode,1)-Eopt;

% 相对最优恢复误差：统一用参考真解的局部泛函值归一化
functional_scale = max(abs(functional_true),eps);

Eopt_rel = Eopt/functional_scale;
Eref_rel = Eref/functional_scale;
gap_rel = gap/functional_scale;

%% ========================================================================
%  8. 输出相对最优恢复误差
% =========================================================================
fprintf('\n============================================================\n');
fprintf('统一真解约束下的相对最优恢复误差 e_n^opt\n');
fprintf('e_n^opt = E_n^opt / |ell(z_true)|\n');
fprintf('n 表示保留 Fourier 模态总数，不是 n x n 网格\n');
fprintf('============================================================\n');

fprintf(' n       delta=.01       delta=.03       delta=.05       delta=.07\n');

for in = 1:n_mode
    fprintf('%3d   %12.8f   %12.8f   %12.8f   %12.8f\n', par.n_mode_list(in), ...
        Eopt_rel(in,1),Eopt_rel(in,2),Eopt_rel(in,3),Eopt_rel(in,4));
end

fprintf('\n256x256 完整参考问题相对最优恢复误差 e_ref^opt：\n');
fprintf('          %12.8f   %12.8f   %12.8f   %12.8f\n', ...
    Eref_rel(1),Eref_rel(2),Eref_rel(3),Eref_rel(4));

fprintf('\n参考归一化因子 |ell(z_true)| = %.12e\n',functional_scale);

%% ========================================================================
%  9. 输出相对有限维逼近误差
% =========================================================================
fprintf('\n============================================================\n');
fprintf('相对有限维逼近误差 e_ref^opt-e_n^opt\n');
fprintf('============================================================\n');

fprintf(' n       delta=.01       delta=.03       delta=.05       delta=.07\n');

for in = 1:n_mode
    fprintf('%3d   %12.4e   %12.4e   %12.4e   %12.4e\n', par.n_mode_list(in), ...
        gap_rel(in,1),gap_rel(in,2),gap_rel(in,3),gap_rel(in,4));
end

fprintf('\n每个有限维子空间的最高保留频率：\n');
for in = 1:n_mode
    fprintf('n=%3d: omega_max = %.6f\n', par.n_mode_list(in),omega_max_used(in));
end

%% ========================================================================
%  10. 数值关系检查
% =========================================================================
% 在统一约束、嵌套子空间下，E_n^opt 应随 n 非减
for id = 1:n_delta

    if any(diff(Eopt(:,id)) < -1.0e-10)
        warning('delta=%.2f 时 E_n^opt 未保持非减，请检查模态选择。',par.delta_list(id));
    end

    if any(Eopt(:,id) > Eref(id)+1.0e-10)
        warning('delta=%.2f 时有限维值超过完整参考值。',par.delta_list(id));
    end
end

% 固定 n 时，噪声增大后最优恢复误差应非减
for in = 1:n_mode
    if any(diff(Eopt(in,:)) < -1.0e-10)
        warning('n=%d 时最优恢复误差未随噪声水平非减。',par.n_mode_list(in));
    end
end

%% ========================================================================
%  11. 图1：相对最优恢复误差
% =========================================================================
figure('Color','w','Name','4.2：有限维相对最优恢复误差');

marker_list = {'o','s','d','^'};

hold on;

for id = 1:n_delta

    plot(par.n_mode_list,Eopt_rel(:,id),'LineWidth',1.6,'Marker',marker_list{id}, ...
        'MarkerSize',7,'DisplayName',sprintf('\\delta=%.2f',par.delta_list(id)));

    % 256x256 完整参考问题最优恢复误差
    yline(Eref_rel(id),'--','LineWidth',1.0,'HandleVisibility','off');
end

hold off; grid on; box on;

xlabel('保留 Fourier 模态总数 n');
ylabel('相对最优恢复误差 e_n^{opt}');
title('统一真解约束下的有限维相对最优恢复误差');

legend('Location','best');

xticks(par.n_mode_list);
xlim([par.n_mode_list(1),par.n_mode_list(end)]);

set(gca,'FontSize',11);

%% ========================================================================
%  12. 图2：相对有限维逼近误差
% =========================================================================
figure('Color','w','Name','4.2：有限维逼近误差');

hold on;

for id = 1:n_delta

    plot(par.n_mode_list,gap_rel(:,id),'LineWidth',1.6,'Marker',marker_list{id}, ...
        'MarkerSize',7,'DisplayName',sprintf('\\delta=%.2f',par.delta_list(id)));
end

hold off; grid on; box on;

xlabel('保留 Fourier 模态总数 n');
ylabel('e_{ref}^{opt}-e_n^{opt}');
title('相对最优恢复误差向 256\times256 参考问题的逼近');

legend('Location','best');

xticks(par.n_mode_list);
xlim([par.n_mode_list(1),par.n_mode_list(end)]);

set(gca,'FontSize',11);

%% ========================================================================
%  固定 Fourier 子空间上的最优恢复误差
% =========================================================================
function E = optimal_recovery_error_masked(Ahat,Rhat,chat,mask,R_prior,Delta_info,par)

    aa = abs(Ahat(mask)).^2;
    rr = Rhat(mask);
    cc = abs(chat(mask)).^2;

    logt_grid = linspace(par.opt_logt_min,par.opt_logt_max,par.opt_grid_size);

    values = zeros(size(logt_grid));

    for k = 1:numel(logt_grid)
        values(k) = optimal_recovery_objective(logt_grid(k),aa,rr,cc,R_prior,Delta_info);
    end

    [coarse_best,idx] = min(values);

    il = max(1,idx-1);
    ir = min(numel(logt_grid),idx+1);

    left = logt_grid(il);
    right = logt_grid(ir);

    objective = @(p) optimal_recovery_objective(p,aa,rr,cc,R_prior,Delta_info);

    if left==right

        best_value = coarse_best;

    else

        [~,best_value] = fminbnd(objective,left,right,optimset( ...
            'Display','off','TolX',par.opt_tol_x,'MaxIter',300));

        best_value = min(best_value,coarse_best);
    end

    % t=0：仅信息误差约束主导
    value_t0 = Delta_info*sqrt(real(sum(cc./aa)));

    % t->Inf：仅先验约束主导
    value_tinf = sqrt(R_prior*real(sum(cc./rr)));

    E = min([best_value,value_t0,value_tinf]);
end

%% ========================================================================
%  最优恢复的一维 Lagrange 目标
% =========================================================================
function value = optimal_recovery_objective(logt,aa,rr,cc,R_prior,Delta_info)

    if ~isfinite(logt)
        value = Inf;
        return;
    end

    t = exp(logt);

    den = aa+t*rr;

    if any(~isfinite(den)) || any(den<=0)
        value = Inf;
        return;
    end

    S = real(sum(cc./den));

    value = sqrt( ...
        (Delta_info^2+t*R_prior)*S );
end

%% ========================================================================
%  Fourier 频率网格
% =========================================================================
function [k2,kabs] = fourier_frequencies_2d(N,L)

    if mod(N,2)==0
        modes = [0:N/2-1,-N/2:-1];
    else
        modes = [0:(N-1)/2,-(N-1)/2:-1];
    end

    k1 = (2*pi/L)*modes;

    [KX,KY] = meshgrid(k1,k1);

    k2 = KX.^2+KY.^2;
    kabs = sqrt(k2);
end

%% ========================================================================
%  离散范数
% =========================================================================
function value = grid_norm(v)
    value = sqrt(mean(abs(v(:)).^2));
end

function value = spectral_omega(zhat,Rhat,N)
    value = real(sum(Rhat(:).*abs(zhat(:)).^2))/N^4;
end
