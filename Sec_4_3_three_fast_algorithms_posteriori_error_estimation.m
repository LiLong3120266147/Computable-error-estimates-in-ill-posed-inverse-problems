% 4.3.2：三种快速算法 全局/局部后验误差估计对比

% 三种方法在前面的所有步骤完全相同:
%   Step 1. Tikhonov + 偏差原则得到 z_delta;
%   Step 2. 仅采用自适应双参数后验可行集合 (C_res^ad,C_Omega^ad);
%   Step 3. 对单位方向 w 计算两个允许步长 t_A(w), t_R(w);
%   Step 4. 只有最后的“合并半径”公式不同;
%   M1  Exact nonsmooth + Clarke subgradient
%       g(w) = min{t_A(w),t_R(w)}.
%   M2  Leonov smooth-min
%       xi_m(a,b) = a*b/(a^m+b^m)^(1/m).
%   M3  Smooth upper-min
%       phi_eps(a,b) = 0.5*(a+b-sqrt((a-b)^2+eps^2)+eps).
% 全局后验估计：
%       E(eta) ~= max_{||w||=1} q(t_A(w),t_R(w)).
% 局部线性泛函后验估计：
%       E_f(eta) ~= max_{||w||=1} |<f,w>| q(t_A(w),t_R(w)).
%
% 其中 q 分别取 M1/M2/M3 的最终半径公式.
% 全局和局部后验估计都由同一个快速算法 3 框架求解; 三种方法只在最后 q(a,b) 处发生分支.
%
% 注意：自适应双参数在本数值实验中利用真解计算使真解进入后验集合所需的
% 最小参数，因此属于数值验证设置，不是实际未知真解情形下可直接计算的参数。

clear; clc; close all;

%% ============================ 0. 参数设置 ============================

cfg.Nfine = 512;
cfg.N = 128;
cfg.zeta = 0.15;
cfg.nu = 0.25;
cfg.deltaList = 0.01:0.01:0.10;

cfg.numRealizations = 1;
cfg.randomSeed = 1218;
cfg.reuseNoiseDirectionAcrossDeltas = true;

% Tikhonov 偏差原则常数
cfg.Cdiscrepancy = 1.01;

% 使用自适应双参数后验可行集合
cfg.Cadaptive.min = 1.01;
cfg.Cadaptive.max = 5.00;
cfg.Cadaptive.safetySteps = 0;

% 正则化泛函 Omega(z)=<z,Rz>
cfg.laplacianScale = 1.0;

% -------------------- M1：Clarke 次梯度参数 --------------------
% 当两个分支足够接近时, 使用 active gradients 的 Clarke convex combination.
cfg.subgradientSwitchTolAbs = 1.0e-12;
cfg.subgradientSwitchTolRel = 1.0e-6;

% -------------------- M2：Leonov 参数 --------------------
cfg.leonovPower = 500;

% -------------------- M3：upper smooth-min 参数 --------------------
% epsilon 与 t_A, t_R 量纲相同.
cfg.upperSmoothEpsilon = 1.0e-4;

% -------------------- 三种方法共用的球面优化参数 --------------------
cfg.maxSphereIterations = 200;
cfg.gradientTolerance = 2.0e-8;
cfg.initialAngleStep = 0.30;
cfg.maxAngleStep = 0.60;
cfg.lineSearchMax = 20;
cfg.smoothArmijo = 1.0e-4;
cfg.subgradientArmijo = 1.0e-6;
cfg.randomStarts = 10;
cfg.verboseOptimizer = false;

% 上一个噪声水平的三种最优方向均加入下一噪声水平的公共 warm starts.
cfg.useSharedWarmStarts = true;

% 显示代表性图像的噪声水平
cfg.showImagesAtDelta = 0.05;

% 局部区域 D1+D2
cfg.region1 = [0.20,0.44,0.20,0.44];
cfg.region2 = [0.45,0.75,0.45,0.75];

methodNames = {'subgradient','leonov','upper'};
methodLabels = {'Exact min + Clarke subgradient','Leonov smooth-min','Upper smooth-min'};
methodTags = {'Subgrad','Leonov','Upper'};
nMethods = numel(methodNames);

fprintf('============================================================\n');
fprintf('Fast Algorithm 3: global + local posterior estimates\n');
fprintf('Posterior set        : adaptive pair only\n');
fprintf('Methods              : subgradient / Leonov / upper smooth\n');
fprintf('Leonov power m       : %d\n',cfg.leonovPower);
fprintf('Upper smooth epsilon : %.3e\n',cfg.upperSmoothEpsilon);
fprintf('============================================================\n');

%% ============================ 1. 构造模型 ============================

fprintf('正在构造模型：Nfine=%d, N=%d ...\n',cfg.Nfine,cfg.N);
model = buildModel_FirstProblem(cfg);
numDelta = numel(cfg.deltaList);

fprintf('局部区域：D1=[%.2f,%.2f]x[%.2f,%.2f], D2=[%.2f,%.2f]x[%.2f,%.2f]\n', ...
    cfg.region1(1),cfg.region1(2),cfg.region1(3),cfg.region1(4), ...
    cfg.region2(1),cfg.region2(2),cfg.region2(3),cfg.region2(4));

%% ============================ 2. 预分配 ============================

trueGlobalAll = NaN(cfg.numRealizations,numDelta);
trueLocalAll = NaN(cfg.numRealizations,numDelta);
alphaAll = NaN(cfg.numRealizations,numDelta);
residualAll = NaN(cfg.numRealizations,numDelta);

CneededDataAll = NaN(cfg.numRealizations,numDelta);
CneededRegAll = NaN(cfg.numRealizations,numDelta);
CresAll = NaN(cfg.numRealizations,numDelta);
ComegaAll = NaN(cfg.numRealizations,numDelta);
feasibleAll = NaN(cfg.numRealizations,numDelta);

% 第三维 = 三种 Algorithm-3 最终半径公式.
globalEstimateAll = NaN(cfg.numRealizations,numDelta,nMethods);
localEstimateAll = NaN(cfg.numRealizations,numDelta,nMethods);

% 仅作为算法诊断: 把每种方法找到的最优方向重新代回原始 min.
% 这不是另一个“原始算法”, 只是评价同一方向时的 exact-min 值.
globalExactMinAtBestAll = NaN(cfg.numRealizations,numDelta,nMethods);
localExactMinAtBestAll = NaN(cfg.numRealizations,numDelta,nMethods);

globalTimeAll = NaN(cfg.numRealizations,numDelta,nMethods);
localTimeAll = NaN(cfg.numRealizations,numDelta,nMethods);
globalIterationsAll = NaN(cfg.numRealizations,numDelta,nMethods);
localIterationsAll = NaN(cfg.numRealizations,numDelta,nMethods);
globalEvaluationsAll = NaN(cfg.numRealizations,numDelta,nMethods);
localEvaluationsAll = NaN(cfg.numRealizations,numDelta,nMethods);

representative = struct();

%% ============================ 3. 主循环 ============================

startTime = tic;

for iReal = 1:cfg.numRealizations

    fprintf('\n============================================================\n');
    fprintf('随机噪声实现 %d / %d\n',iReal,cfg.numRealizations);
    fprintf('============================================================\n');

    rng(cfg.randomSeed+iReal-1,'twister');

    if cfg.reuseNoiseDirectionAcrossDeltas
        commonNoiseFine = randn(cfg.Nfine,cfg.Nfine);
        commonNoiseFine = commonNoiseFine/l2norm(commonNoiseFine);
        commonNoiseCoarse = commonNoiseFine( ...
            model.fineToCoarseIndex,model.fineToCoarseIndex);
        commonNoiseCoarseNorm = l2norm(commonNoiseCoarse);
        if commonNoiseCoarseNorm <= eps
            error('限制到粗网格后的随机噪声方向范数过小。');
        end
    end

    previousGlobalBest = cell(1,nMethods);
    previousLocalBest = cell(1,nMethods);

    for id = 1:numDelta

        delta = cfg.deltaList(id);

        %% ---------- Step 1. 含噪数据 + Tikhonov + 偏差原则 ----------
        if cfg.reuseNoiseDirectionAcrossDeltas
            noiseScale = delta*l2norm(model.uExact)/commonNoiseCoarseNorm;
            noiseFine = noiseScale*commonNoiseFine;
            uDeltaFine = model.uFine+noiseFine;
            uDelta = uDeltaFine(modelfinder.fineToCoarseIndex,model.fineToCoarseIndex);
            noise = uDelta-model.uExact;
        else
            noiseDirection = normalizeL2(randn(cfg.N,cfg.N));
            noise = delta*l2norm(model.uExact)*noiseDirection;
            uDelta = model.uExact+noise;
        end

        noiseNorm = l2norm(noise);
        [zApprox,alpha,residualNorm] = solveTikhonovDiscrepancy(uDelta,model,noiseNorm,cfg);

        alphaAll(iReal,id) = alpha;
        residualAll(iReal,id) = residualNorm;

        normZExact = max(l2norm(model.zExact),eps);
        exactFunctional = innerL2(model.fWeight,model.zExact);
        absExactFunctional = max(abs(exactFunctional),eps);

        % 真实误差用于验证三种后验估计是否覆盖
        trueGlobalAll(iReal,id) = l2norm(zApprox-model.zExact)/normZExact;
        trueLocalAll(iReal,id) = abs(innerL2(model.fWeight,zApprox-model.zExact))/absExactFunctional;

        %% ---------- Step 2. 自适应双参数后验可行集合 ----------
        AzExact = applyMultiplier(model.zExact,model.Ahat);
        RzExact = applyMultiplier(model.zExact,model.Rhat);
        RzApprox = applyMultiplier(zApprox,model.Rhat);

        OmegaExact = max(innerL2(model.zExact,RzExact),0);
        OmegaApprox = max(innerL2(zApprox,RzApprox),eps);
        trueResidual = l2norm(AzExact-uDelta);

        CneededData = trueResidual/max(residualNorm,eps);
        CneededReg = OmegaExact/max(OmegaApprox,eps);
        [Cres,Comega] = adaptivePosteriorPair(cfg,CneededData,CneededReg);

        CneededDataAll(iReal,id) = CneededData;
        CneededRegAll(iReal,id) = CneededReg;
        CresAll(iReal,id) = Cres;
        ComegaAll(iReal,id) = Comega;
        feasibleAll(iReal,id) = double(Cres >= CneededData-1e-12 && Comega >= CneededReg-1e-12);

        %% ---------- Step 3. 公共 t_A(w), t_R(w) 问题 ----------
        % 全局和局部、三种方法都共用同一个 posterior 结构.
        posterior = buildPosteriorProblem( zApprox,uDelta,model,Cres,Comega);

        %% ---------- Step 4-5A. 快速算法 3：全局后验估计 ----------
        startsGlobal = buildStartDirections(zApprox,uDelta,model,cfg,model.emptyWeight);
        if cfg.useSharedWarmStarts
            startsGlobal = addSharedWarmStarts(startsGlobal,previousGlobalBest);
        end

        globalOut = cell(1,nMethods);
        for im = 1:nMethods
            globalOut{im} = maximizeOnL2SphereMethod(posterior,startsGlobal,[],cfg,methodNames{im});
            out = globalOut{im};

            % Algorithm 3 对应方法的最终公式值.
            globalEstimateAll(iReal,id,im) = out.surrogateValue/normZExact;

            % 仅诊断：同一最优方向代回 exact min.
            globalExactMinAtBestAll(iReal,id,im) = out.originalValue/normZExact;

            globalTimeAll(iReal,id,im) = out.elapsedTime;
            globalIterationsAll(iReal,id,im) = out.totalIterations;
            globalEvaluationsAll(iReal,id,im) = out.functionEvaluations;
            previousGlobalBest{im} = out.bestW;
        end

        %% ---------- Step 4-5B. 快速算法 3：局部后验估计 ----------
        startsLocal = buildStartDirections(zApprox,uDelta,model,cfg,model.fWeight);
        if cfg.useSharedWarmStarts
            startsLocal = addSharedWarmStarts(startsLocal,previousLocalBest);
        end

        localOut = cell(1,nMethods);
        for im = 1:nMethods
            localOut{im} = maximizeOnL2SphereMethod(posterior,startsLocal,model.fWeight,cfg,methodNames{im});
            out = localOut{im};

            localEstimateAll(iReal,id,im) = out.surrogateValue/absExactFunctional;
            localExactMinAtBestAll(iReal,id,im) = out.originalValue/absExactFunctional;

            localTimeAll(iReal,id,im) = out.elapsedTime;
            localIterationsAll(iReal,id,im) = out.totalIterations;
            localEvaluationsAll(iReal,id,im) = out.functionEvaluations;
            previousLocalBest{im} = out.bestW;
        end

        %% ---------- 输出 ----------
        fprintf(['delta=%5.2f | alpha=%9.3e | res=%8.2e | ','C_ad=(%.2f,%.2f) | trueG=%7.4f trueL=%7.4f\n'], ...
            delta,alpha,residualNorm,Cres,Comega,trueGlobalAll(iReal,id),trueLocalAll(iReal,id));

        for im = 1:nMethods
            fprintf(['  %-34s ','Global=%8.5f  Local=%8.5f | ','G exact-min@best=%8.5f  L exact-min@best=%8.5f | ', ...
                'time=(%.2fs, %.2fs)\n'],methodLabels{im},globalEstimateAll(iReal,id,im),localEstimateAll(iReal,id,im), ...
                globalExactMinAtBestAll(iReal,id,im),localExactMinAtBestAll(iReal,id,im), globalTimeAll(iReal,id,im), ...
                localTimeAll(iReal,id,im));
        end

        %% ---------- 保存代表性结果 ----------
        if iReal==1 && abs(delta-cfg.showImagesAtDelta)<1e-12
            representative.delta = delta;
            representative.uDelta = uDelta;
            representative.zApprox = zApprox;
            representative.error = zApprox-model.zExact;
            representative.globalCandidate = cell(1,nMethods);
            for im = 1:nMethods
                out = globalOut{im};
                % 用原始可行半径生成实际可行的最坏候选解.
                tFeasible = min(out.tA,out.tR);
                representative.globalCandidate{im} = zApprox+tFeasible*out.bestW;
            end
        end

    end
end

elapsedTime = toc(startTime);

%% ============================ 4. 平均结果 ============================

trueGlobal = mean(trueGlobalAll,1,'omitnan');
trueLocal = mean(trueLocalAll,1,'omitnan');
alphaMean = mean(alphaAll,1,'omitnan');
residualMean = mean(residualAll,1,'omitnan');
CneededDataMean = mean(CneededDataAll,1,'omitnan');
CneededRegMean = mean(CneededRegAll,1,'omitnan');
CresMean = mean(CresAll,1,'omitnan');
ComegaMean = mean(ComegaAll,1,'omitnan');
feasibleRate = mean(feasibleAll,1,'omitnan');

globalEstimate = squeeze(mean(globalEstimateAll,1,'omitnan'));
localEstimate = squeeze(mean(localEstimateAll,1,'omitnan'));
globalExactMinAtBest = squeeze(mean(globalExactMinAtBestAll,1,'omitnan'));
localExactMinAtBest = squeeze(mean(localExactMinAtBestAll,1,'omitnan'));
globalTime = squeeze(mean(globalTimeAll,1,'omitnan'));
localTime = squeeze(mean(localTimeAll,1,'omitnan'));
globalIterations = squeeze(mean(globalIterationsAll,1,'omitnan'));
localIterations = squeeze(mean(localIterationsAll,1,'omitnan'));
globalEvaluations = squeeze(mean(globalEvaluationsAll,1,'omitnan'));
localEvaluations = squeeze(mean(localEvaluationsAll,1,'omitnan'));

if numDelta==1
    globalEstimate = reshape(globalEstimate,1,nMethods);
    localEstimate = reshape(localEstimate,1,nMethods);
    globalExactMinAtBest = reshape(globalExactMinAtBest,1,nMethods);
    localExactMinAtBest = reshape(localExactMinAtBest,1,nMethods);
    globalTime = reshape(globalTime,1,nMethods);
    localTime = reshape(localTime,1,nMethods);
    globalIterations = reshape(globalIterations,1,nMethods);
    localIterations = reshape(localIterations,1,nMethods);
    globalEvaluations = reshape(globalEvaluations,1,nMethods);
    localEvaluations = reshape(localEvaluations,1,nMethods);
end

resultTable = table(cfg.deltaList(:),alphaMean(:),residualMean(:),trueGlobal(:),trueLocal(:), ...
    CresMean(:),ComegaMean(:),CneededDataMean(:),CneededRegMean(:),feasibleRate(:), ...
    'VariableNames',{'delta','alpha','residual','trueGlobal','trueLocal','CresAdaptive', ...
    'ComegaAdaptive','CneededData','CneededRegularizer','adaptivePairFeasibleRate'});

for im = 1:nMethods
    tag = methodTags{im};

    resultTable.(sprintf('globalAlg3_%s',tag)) = globalEstimate(:,im);
    resultTable.(sprintf('localAlg3_%s',tag)) = localEstimate(:,im);

    resultTable.(sprintf('globalExactMinAtBest_%s',tag)) = globalExactMinAtBest(:,im);
    resultTable.(sprintf('localExactMinAtBest_%s',tag)) = localExactMinAtBest(:,im);

    resultTable.(sprintf('globalTime_%s',tag)) = globalTime(:,im);
    resultTable.(sprintf('localTime_%s',tag)) = localTime(:,im);
    resultTable.(sprintf('globalIterations_%s',tag)) = globalIterations(:,im);
    resultTable.(sprintf('localIterations_%s',tag)) = localIterations(:,im);
    resultTable.(sprintf('globalEvaluations_%s',tag)) = globalEvaluations(:,im);
    resultTable.(sprintf('localEvaluations_%s',tag)) = localEvaluations(:,im);
end

fprintf('\n======================== 平均结果 ========================\n');
disp(resultTable);
fprintf('总运行时间：%.2f s\n',elapsedTime);

%% ============================ 5. 作图 ============================

methodColors = lines(nMethods);
methodMarkers = {'o','s','d'};

% -------------------- 图1：全局后验误差估计 --------------------
figure('Name','Algorithm 3 - Global posterior','Color','w');
plot(cfg.deltaList,trueGlobal,'k^-','LineWidth',1.6,'MarkerSize',6,'DisplayName','近似解真实全局误差');
hold on;
for im = 1:nMethods
    plot(cfg.deltaList,globalEstimate(:,im),[methodMarkers{im} '-'],'Color',methodColors(im,:), ...
        'LineWidth',1.5,'MarkerSize',6,'DisplayName',methodLabels{im});
end
hold off; grid on; box on;
xlabel('相对噪声水平 \delta');
ylabel('相对误差');
title('快速算法 3：全局后验误差估计');
legend('Location','northwest','Interpreter','none');
xlim([cfg.deltaList(1),cfg.deltaList(end)]);
xticks(cfg.deltaList);

% -------------------- 图2：局部后验误差估计 --------------------
figure('Name','Algorithm 3 - Local posterior','Color','w');
plot(cfg.deltaList,trueLocal,'k^-','LineWidth',1.6,'MarkerSize',6,'DisplayName','近似解真实局部误差');
hold on;
for im = 1:nMethods
    plot(cfg.deltaList,localEstimate(:,im),[methodMarkers{im} '-'],'Color',methodColors(im,:), ...
        'LineWidth',1.5,'MarkerSize',6,'DisplayName',methodLabels{im});
end
hold off; grid on; box on;
xlabel('相对噪声水平 \delta');
ylabel('相对误差');
title('快速算法 3：D_1+D_2 局部后验误差估计');
legend('Location','northwest','Interpreter','none');
xlim([cfg.deltaList(1),cfg.deltaList(end)]);
xticks(cfg.deltaList);

% -------------------- 图3：三种最终公式在各自最优方向上的偏差 --------------------
figure('Name','Algorithm 3 - Final formula discrepancy','Color','w');
subplot(1,2,1);
for im = 1:nMethods
    gap = (globalEstimate(:,im)-globalExactMinAtBest(:,im))./max(globalExactMinAtBest(:,im),eps);
    plot(cfg.deltaList,gap,[methodMarkers{im} '-'],'Color',methodColors(im,:), ...
        'LineWidth',1.4,'MarkerSize',5,'DisplayName',methodLabels{im});
    hold on;
end
yline(0,'k--'); hold off; grid on; box on;
xlabel('\delta');
ylabel('(method value - exact min)/exact min');
title('Global final-formula discrepancy');
legend('Location','best','Interpreter','none');

subplot(1,2,2);
for im = 1:nMethods
    gap = (localEstimate(:,im)-localExactMinAtBest(:,im))./max(localExactMinAtBest(:,im),eps);
    plot(cfg.deltaList,gap,[methodMarkers{im} '-'],'Color',methodColors(im,:), ...
        'LineWidth',1.4,'MarkerSize',5,'DisplayName',methodLabels{im});
    hold on;
end
yline(0,'k--'); hold off; grid on; box on;
xlabel('\delta');
ylabel('(method value - exact min)/exact min');
title('Local final-formula discrepancy');
legend('Location','best','Interpreter','none');

% -------------------- 图4：运行时间 --------------------
figure('Name','Algorithm 3 - Runtime','Color','w');
subplot(1,2,1);
for im = 1:nMethods
    plot(cfg.deltaList,globalTime(:,im),[methodMarkers{im} '-'],'Color',methodColors(im,:), ...
        'LineWidth',1.4,'MarkerSize',5,'DisplayName',methodLabels{im});
    hold on;
end
hold off; grid on; box on;
xlabel('\delta'); ylabel('CPU time (s)');
title('Global Algorithm-3 optimizer');
legend('Location','best','Interpreter','none');

subplot(1,2,2);
for im = 1:nMethods
    plot(cfg.deltaList,localTime(:,im),[methodMarkers{im} '-'],'Color',methodColors(im,:), ...
        'LineWidth',1.4,'MarkerSize',5,'DisplayName',methodLabels{im});
    hold on;
end
hold off; grid on; box on;
xlabel('\delta'); ylabel('CPU time (s)');
title('Local Algorithm-3 optimizer');
legend('Location','best','Interpreter','none');

% -------------------- 图5：自适应双参数 --------------------
figure('Name','Adaptive posterior pair','Color','w');
plot(cfg.deltaList,CneededDataMean,'v-','LineWidth',1.4,'MarkerSize',5,'DisplayName','C_{res}^{needed}');
hold on;
plot(cfg.deltaList,CneededRegMean,'o-','LineWidth',1.4,'MarkerSize',5,'DisplayName','C_{\Omega}^{needed}');
plot(cfg.deltaList,CresMean,'--','LineWidth',1.4,'DisplayName','C_{res}^{ad}');
plot(cfg.deltaList,ComegaMean,'-.','LineWidth',1.4,'DisplayName','C_{\Omega}^{ad}');
hold off; grid on; box on;
xlabel('\delta'); ylabel('coefficient');
title('自适应双参数后验可行集合');
legend('Location','best','Interpreter','tex');

% -------------------- 图6：精确解和局部区域 --------------------
figure('Name','Exact solution and local rectangles','Color','w');
imagesc(model.x,model.x,model.zExact);
axis image xy; colorbar; hold on;
rectangle('Position',[cfg.region1(1),cfg.region1(3),cfg.region1(2)-cfg.region1(1), ...
    cfg.region1(4)-cfg.region1(3)],'EdgeColor','w','LineWidth',1.5,'LineStyle','--');
rectangle('Position',[cfg.region2(1),cfg.region2(3),cfg.region2(2)-cfg.region2(1), ...
    cfg.region2(4)-cfg.region2(3)],'EdgeColor','w','LineWidth',1.5,'LineStyle','--');
hold off;
xlabel('x_1'); ylabel('x_2');
title('Exact field and D_1+D_2 rectangles');

% -------------------- 图7：代表性全局最坏候选解 --------------------
if isfield(representative,'globalCandidate')
    figure('Name','Representative global candidates','Color','w');
    subplot(2,3,1);
    imagesc(model.x,model.x,model.zExact);
    axis image xy; colorbar;
    title('Exact z');

    subplot(2,3,2);
    imagesc(model.x,model.x,representative.zApprox);
    axis image xy; colorbar;
    title(sprintf('Tikhonov, \\delta=%.2f',representative.delta));

    subplot(2,3,3);
    imagesc(model.x,model.x,representative.error);
    axis image xy; colorbar;
    title('Tikhonov error');

    for im = 1:nMethods
        subplot(2,3,3+im);
        imagesc(model.x,model.x,representative.globalCandidate{im});
        axis image xy; colorbar;
        title(methodLabels{im},'Interpreter','none');
    end
end

%% ============================ 局部函数 ============================

function model = buildModel_FirstProblem(cfg)

    if mod(cfg.Nfine,cfg.N)~=0
        error('cfg.Nfine 必须是 cfg.N 的整数倍。');
    end

    % 细网格信号
    xFine = (0:cfg.Nfine-1)/cfg.Nfine;
    [Xf,Yf] = meshgrid(xFine,xFine);
    T1fine = (Xf-0.32).^2 + (Yf-0.32).^2 < 0.0004;
    T2fine = (Xf-0.60).^2 - (Xf-0.60).*(Yf-0.60) + (Yf-0.60).^2 < 0.01;
    sourceFine = 20*double(T1fine) + 1*double(T2fine);
    [~,omega2Fine] = frequencyGrid(cfg.Nfine);
    kabsFine = sqrt(omega2Fine);

    % 周期 Fourier Poisson 延拓
    zFineHat = exp(-cfg.zeta*kabsFine).*fft2(sourceFine);
    zFine = real(ifft2(zFineHat));
    uFine = real(ifft2(exp(-(cfg.nu-cfg.zeta)*kabsFine).*zFineHat));
    stride = cfg.Nfine/cfg.N;
    index = 1:stride:cfg.Nfine;
    zExact = zFine(index,index);
    uExact = uFine(index,index);

    % 粗网格信息
    x = (0:cfg.N-1)/cfg.N;
    [X,Y] = meshgrid(x,x);
    T1 = (X-0.32).^2 + (Y-0.32).^2 < 0.0004;
    T2 = (X-0.60).^2 - (X-0.60).*(Y-0.60) + (Y-0.60).^2 < 0.01;
    sourceCoarse = 20*double(T1) + 1*double(T2);
    [~,omega2] = frequencyGrid(cfg.N);
    kabs = sqrt(omega2);
    Ahat = exp(-(cfg.nu-cfg.zeta)*kabs);
    Rhat = 1 + (cfg.laplacianScale*omega2).^2;

    % D1+D2 矩形区域
    b1 = cfg.region1;
    b2 = cfg.region2;
    mask1 = X>=b1(1) & X<=b1(2) & Y>=b1(3) & Y<=b1(4);
    mask2 = X>=b2(1) & X<=b2(2) & Y>=b2(3) & Y<=b2(4);
    mask = mask1 | mask2;
    if ~any(mask(:))
        error('D1+D2 矩形区域为空，请检查 cfg.region1 和 cfg.region2。');
    end

    % innerL2(fWeight,z) = D1+D2 并集上的区域平均值.
    fWeight = double(mask)/mean(double(mask(:)));
    model.x = x;
    model.zExact = zExact;
    model.uExact = uExact;
    model.uFine = uFine;
    model.fineToCoarseIndex = index;
    model.sourceCoarse = sourceCoarse;
    model.fWeight = fWeight;
    model.localMask = mask;
    model.localMask1 = mask1;
    model.localMask2 = mask2;
    model.Ahat = Ahat;
    model.A2hat = Ahat.^2;
    model.Rhat = Rhat;
    model.omega2 = omega2;
    model.emptyWeight = [];
end

function [k,omega2] = frequencyGrid(N)
    if mod(N,2)~=0
        error('本程序要求 N 为偶数。');
    end

    modes = [0:N/2-1,-N/2:-1];
    [KX,KY] = meshgrid(modes,modes);
    omega2 = (2*pi)^2*(KX.^2+KY.^2);
    k = sqrt(KX.^2+KY.^2);
end

function y = applyMultiplier(x,multiplier)
    y = real(ifft2(multiplier.*fft2(x)));
end

function [zApprox,alpha,residualNorm] = solveTikhonovDiscrepancy(uDelta,model,noiseNorm,cfg)
    
    target = cfg.Cdiscrepancy*noiseNorm;
    uHat = fft2(uDelta);
    alphaLeft = 1e-20;
    alphaRight = 1e2;
    residualRight = residualAt(alphaRight,uHat,uDelta,model);

    while residualRight < target
        alphaRight = 10*alphaRight;

        if alphaRight > 1e50
            error('无法括住偏差原则参数 alpha。');
        end

        residualRight = residualAt(alphaRight,uHat,uDelta,model);
    end

    for iter = 1:60
        alphaMid = sqrt(alphaLeft*alphaRight);
        residualMid = residualAt(alphaMid,uHat,uDelta,model);

        if residualMid < target
            alphaLeft = alphaMid;
        else
            alphaRight = alphaMid;
        end
    end

    alpha = sqrt(alphaLeft*alphaRight);
    zApprox = tikhonovAt(alpha,uHat,model);
    residualNorm = l2norm(applyMultiplier(zApprox,model.Ahat)-uDelta);
end

function value = residualAt(alpha,uHat,uDelta,model)
    z = tikhonovAt(alpha,uHat,model);
    value = l2norm(applyMultiplier(z,model.Ahat)-uDelta);
end

function z = tikhonovAt(alpha,uHat,model)
    zHat = conj(model.Ahat).*uHat./(abs(model.Ahat).^2 + alpha*model.Rhat);
    z = real(ifft2(zHat));
end



function [Cres,Comega] = adaptivePosteriorPair(cfg,CneededData,CneededReg)
    % 自适应双参数: 分别保证真解满足残差约束和稳定化约束.
    Cres = ceil_to_two_decimals(max(cfg.Cadaptive.min,CneededData));
    Comega = ceil_to_two_decimals(max(cfg.Cadaptive.min,CneededReg));

    Cres = min(max(Cres,cfg.Cadaptive.min),cfg.Cadaptive.max);
    Comega = min(max(Comega,cfg.Cadaptive.min),cfg.Cadaptive.max);

    Cres = Cres+0.01*cfg.Cadaptive.safetySteps;
    Comega = Comega+0.01*cfg.Cadaptive.safetySteps;

    if Cres < 1 || Comega < 1
        warning('当前自适应后验系数小于 1，可能导致方向允许半径为空或退化。');
    end
end

function posterior = buildPosteriorProblem(zApprox,uDelta,model,Cres,Comega)

    v = uDelta-applyMultiplier(zApprox,model.Ahat);

    posterior.Ahat = model.Ahat;
    posterior.A2hat = model.A2hat;
    posterior.Rhat = model.Rhat;
    posterior.v = v;
    posterior.Atv = applyMultiplier(v,conj(model.Ahat));
    posterior.Rz = applyMultiplier(zApprox,model.Rhat);

    % ||A(zApprox+t w)-uDelta|| <= Cres ||AzApprox-uDelta||
    posterior.rA2 = max((Cres^2-1)*l2norm(v)^2,0);

    % Omega(zApprox+t w) <= Comega Omega(zApprox)
    posterior.rR2 = max((Comega-1)*innerL2(zApprox,posterior.Rz),0);
end

function starts = buildStartDirections(zApprox,uDelta,model,cfg,localWeight)

    residual = uDelta-applyMultiplier(zApprox,model.Ahat);
    AtResidual = applyMultiplier(residual,conj(model.Ahat));
    smoothResidual = applyMultiplier(AtResidual,1./sqrt(model.Rhat));

    starts = {zApprox,-zApprox,AtResidual,-AtResidual,smoothResidual,-smoothResidual};

    if ~isempty(localWeight)
        starts = [starts,{localWeight,-localWeight}]; %#ok<AGROW>
    end

    oldState = rng;
    smoothScales = [0.01 0.03 0.06 0.10];

    for j = 1:cfg.randomStarts
        q = randn(size(zApprox));
        filter = exp(-smoothScales(1+mod(j-1,numel(smoothScales)))*sqrt(model.omega2));
        q = applyMultiplier(q,filter);
        starts = [starts,{q,-q}]; %#ok<AGROW>
    end

    rng(oldState);

    keep = true(size(starts));
    for j = 1:numel(starts)
        keep(j) = l2norm(starts{j})>1e-14;
    end
    starts = starts(keep);
end

function starts = addSharedWarmStarts(starts,previousBest)
    for j = 1:numel(previousBest)
        if ~isempty(previousBest{j})
            starts = [starts,{previousBest{j},-previousBest{j}}]; %#ok<AGROW>
        end
    end
end

function out = maximizeOnL2SphereMethod(posterior,starts,localWeight,cfg,method)
    timer = tic;
    bestScore = -inf;
    bestW = normalizeL2(starts{1});
    bestInfo = [];
    bestIterations = 0;
    totalIterations = 0;
    functionEvaluations = 0;

    for startIndex = 1:numel(starts)

        w = normalizeL2(starts{startIndex});
        info = evaluateMethodObjective(w,posterior,localWeight,cfg,method,true);
        functionEvaluations = functionEvaluations+1;

        % 局部泛函如果恰好与初值正交, 轻微扰动, 避免 log|<f,w>| 的奇点.
        if ~isempty(localWeight) && abs(info.fw)<1e-14
            w = normalizeL2(w+1e-2*localWeight);
            info = evaluateMethodObjective(w,posterior,localWeight,cfg,method,true);
            functionEvaluations = functionEvaluations+1;
        end

        angleStep = cfg.initialAngleStep;
        iterUsed = 0;

        for iter = 1:cfg.maxSphereIterations

            iterUsed = iter;
            gradient = info.gradLogObjective;
            gradient = gradient-innerL2(w,gradient)*w;
            gradientNorm = l2norm(gradient);

            if ~isfinite(gradientNorm) || gradientNorm<cfg.gradientTolerance
                break;
            end

            direction = gradient/gradientNorm;
            trialAngle = min(angleStep,cfg.maxAngleStep);
            accepted = false;

            if strcmpi(method,'subgradient')
                armijo = cfg.subgradientArmijo;
            else
                armijo = cfg.smoothArmijo;
            end

            for lineIter = 1:cfg.lineSearchMax

                wTrial = cos(trialAngle)*w+sin(trialAngle)*direction;
                trial = evaluateMethodObjective( ...
                    wTrial,posterior,localWeight,cfg,method,false);
                functionEvaluations = functionEvaluations+1;

                requiredLogIncrease = armijo*trialAngle*gradientNorm;

                if trial.logObjective >= info.logObjective+requiredLogIncrease
                    accepted = true;
                    break;
                end

                % 对非光滑次梯度, 在切换面附近允许“单调不降”作为回退条件.
                if strcmpi(method,'subgradient') && ...
                        trial.logObjective >= info.logObjective-1e-13
                    accepted = true;
                    break;
                end

                trialAngle = 0.5*trialAngle;
            end

            if ~accepted
                break;
            end

            w = wTrial;
            info = evaluateMethodObjective(w,posterior,localWeight,cfg,method,true);
            functionEvaluations = functionEvaluations+1;

            angleStep = min(cfg.maxAngleStep,1.5*trialAngle);

            if trialAngle*gradientNorm<cfg.gradientTolerance
                break;
            end
        end

        totalIterations = totalIterations+iterUsed;

        % 三种方法均按各自 Algorithm-3 最终公式选择最佳起点.
        candidateScore = info.surrogateObjective;

        if cfg.verboseOptimizer
            fprintf('    %-11s start %2d: original=%9.3e surrogate=%9.3e iter=%d\n', ...
                method,startIndex,info.originalObjective,info.surrogateObjective,iterUsed);
        end

        if candidateScore>bestScore
            bestScore = candidateScore;
            bestW = w;
            bestInfo = info;
            bestIterations = iterUsed;
        end
    end

    % 用 bestW 再精确评价一次, 避免返回过程中遗留旧变量.
    bestInfo = evaluateMethodObjective(bestW,posterior,localWeight,cfg,method,false);
    functionEvaluations = functionEvaluations+1;

    out = struct();
    out.method = method;
    out.bestW = bestW;
    out.originalValue = bestInfo.originalObjective;
    out.surrogateValue = bestInfo.surrogateObjective;
    out.originalRadius = bestInfo.originalRadius;
    out.surrogateRadius = bestInfo.surrogateRadius;
    out.tA = bestInfo.tA;
    out.tR = bestInfo.tR;
    out.fw = bestInfo.fw;
    out.bestIterations = bestIterations;
    out.totalIterations = totalIterations;
    out.functionEvaluations = functionEvaluations;
    out.elapsedTime = toc(timer);
end

function info = evaluateMethodObjective(w,posterior,localWeight,cfg,method,needGradient)

    branches = evaluateRadiusBranches(w,posterior,needGradient);

    tA = branches.tA;
    tR = branches.tR;
    originalRadius = min(tA,tR);

    blendTheta = NaN;

    switch lower(method)

        case 'subgradient'
            surrogateRadius = originalRadius;
            logSurrogateRadius = log(max(surrogateRadius,realmin));

            if needGradient
                gA = branches.gradTA/tA;
                gR = branches.gradTR/tR;
                switchTol = cfg.subgradientSwitchTolAbs+ ...
                    cfg.subgradientSwitchTolRel*max([tA,tR,realmin]);

                if tA<tR-switchTol
                    gradLogRadius = gA;
                    blendTheta = 1;
                elseif tR<tA-switchTol
                    gradLogRadius = gR;
                    blendTheta = 0;
                else
                    % Clarke subdifferential conv{gA,gR}.
                    % 在球面切空间中选择最小范数 convex combination.
                    pA = gA-innerL2(w,gA)*w;
                    pR = gR-innerL2(w,gR)*w;
                    d = pA-pR;
                    dd = innerL2(d,d);
                    if dd<=1e-30
                        blendTheta = 0.5;
                    else
                        blendTheta = -innerL2(pR,d)/dd;
                        blendTheta = min(max(blendTheta,0),1);
                    end
                    gradLogRadius = blendTheta*gA+(1-blendTheta)*gR;
                end
            else
                gradLogRadius = [];
            end

        case 'leonov'
            m = cfg.leonovPower;
            if ~(isscalar(m) && isfinite(m) && m>0)
                error('cfg.leonovPower 必须为正数。');
            end

            logTA = log(tA);
            logTR = log(tR);
            maxLog = max(logTA,logTR);
            expA = exp(max(m*(logTA-maxLog),-745));
            expR = exp(max(m*(logTR-maxLog),-745));
            denominator = expA+expR;
            weightA = expA/denominator;
            weightR = expR/denominator;

            logSurrogateRadius = logTA+logTR-maxLog-log(denominator)/m;
            surrogateRadius = exp(logSurrogateRadius);

            if needGradient
                gradLogRadius = weightR*(branches.gradTA/tA)+ ...
                    weightA*(branches.gradTR/tR);
            else
                gradLogRadius = [];
            end

        case 'upper'
            epsSmooth = cfg.upperSmoothEpsilon;
            if ~(isscalar(epsSmooth) && isfinite(epsSmooth) && epsSmooth>0)
                error('cfg.upperSmoothEpsilon 必须为有限正数。');
            end

            diffTR = tA-tR;
            absDiff = abs(diffTR);
            rootDiff = hypot(diffTR,epsSmooth);

            % 稳定计算
            correction = 0.5*(epsSmooth-epsSmooth^2/(absDiff+rootDiff));
            surrogateRadius = min(tA,tR)+correction;
            surrogateRadius = max(surrogateRadius,realmin);
            logSurrogateRadius = log(surrogateRadius);

            if needGradient
                weightTA = 0.5*(1-diffTR/rootDiff);
                weightTR = 0.5*(1+diffTR/rootDiff);
                gradRadius = weightTA*branches.gradTA+weightTR*branches.gradTR;
                gradLogRadius = gradRadius/surrogateRadius;
            else
                gradLogRadius = [];
            end

        otherwise
            error('未知 method=%s。',method);
    end

    if isempty(localWeight)
        fw = 1;
        originalObjective = originalRadius;
        surrogateObjective = surrogateRadius;
        logObjective = logSurrogateRadius;
        if needGradient
            gradLogObjective = gradLogRadius;
        else
            gradLogObjective = [];
        end
    else
        fw = innerL2(localWeight,w);
        absFw = abs(fw);
        originalObjective = originalRadius*absFw;
        surrogateObjective = surrogateRadius*absFw;
        logObjective = logSurrogateRadius+log(absFw+realmin);
        if needGradient
            if abs(fw)<1e-14
                gradLogObjective = gradLogRadius;
            else
                gradLogObjective = gradLogRadius+localWeight/fw;
            end
        else
            gradLogObjective = [];
        end
    end

    info = struct();
    info.tA = tA;
    info.tR = tR;
    info.originalRadius = originalRadius;
    info.surrogateRadius = surrogateRadius;
    info.originalObjective = originalObjective;
    info.surrogateObjective = surrogateObjective;
    info.logObjective = logObjective;
    info.gradLogObjective = gradLogObjective;
    info.fw = fw;
    info.blendTheta = blendTheta;
end

function branches = evaluateRadiusBranches(w,posterior,needGradient)

    W = fft2(w);
    Aw = real(ifft2(posterior.Ahat.*W));
    Rw = real(ifft2(posterior.Rhat.*W));

    a2 = max(innerL2(Aw,Aw),realmin);
    b = innerL2(Aw,posterior.v);
    sqrtDA = sqrt(max(b^2+a2*posterior.rA2,realmin));
    tA = max((b+sqrtDA)/a2,realmin);

    r2 = max(innerL2(w,Rw),realmin);
    c = innerL2(w,posterior.Rz);
    sqrtDR = sqrt(max(c^2+r2*posterior.rR2,realmin));
    tR = max((-c+sqrtDR)/r2,realmin);

    branches = struct();
    branches.tA = tA;
    branches.tR = tR;

    if needGradient
        A2w = real(ifft2(posterior.A2hat.*W));
        gradTA = tA*(posterior.Atv-tA*A2w)/sqrtDA;
        gradTR = -tR*(posterior.Rz+tR*Rw)/sqrtDR;
        branches.gradTA = gradTA;
        branches.gradTR = gradTR;
    else
        branches.gradTA = [];
        branches.gradTR = [];
    end
end


function value = innerL2(x,y)
    value = mean(real(x(:).*y(:)));
end

function value = l2norm(x)
    value = sqrt(max(innerL2(x,x),0));
end

function x = normalizeL2(x)
    n = l2norm(x);

    if n <= 1e-15
        error('归一化失败：方向范数过小。');
    end
    
    x = x/n;
end

function value = ceil_to_two_decimals(x)
    scaled = 100*x;
    nearestInteger = round(scaled);
    tol = 1e-10*max(1,abs(scaled));

    if abs(scaled-nearestInteger)<=tol
        value = nearestInteger/100;
    else
        value = ceil(scaled)/100;
    end
end
