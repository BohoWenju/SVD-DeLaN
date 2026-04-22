%% Flexible resin 80 A
close all
clear all
clc

%material = readtable('DBPM_FT5000_r0_5_3.dat');

Arruda_Boyce = load('Arruda-Boyce.txt');
Ogden = load('Ogden.txt');
Mooney_Rivlin = load('Mooney-Rivlin.txt');
Neo_Hooke = load('Neo-Hookean.txt');
Yeoh = load('Yeoh.txt');

figure
plot(Yeoh(:,1)*100, Yeoh(:,2), 'or')
hold on
plot(Arruda_Boyce(:,1)*100, Arruda_Boyce(:,3), 'b', 'LineWidth',2)
hold on
plot(Ogden(:,1)*100, Ogden(:,3), 'g', 'LineWidth',2)
hold  on 
plot(Neo_Hooke(:,1)*100, Neo_Hooke(:,3), 'm', 'LineWidth',1)
hold on 
plot(Mooney_Rivlin(:,1)*100, Mooney_Rivlin(:,3), 'k', 'LineWidth',2)
% hold on
% plot(Yeoh(:,1)*100, Yeoh(:,3), 'c', 'LineWidth',2) %% not existing in SOFA
title("\textbf{b) Hyperelastic curve fitting}",'Interpreter','latex','FontSize',30)
ylabel('\textbf{Stress [MPa]}','Interpreter','latex','FontSize',25)
xlabel('\textbf{Strain [$\%$]}','Interpreter','latex','FontSize',25)
ax = gca;
set(ax,'FontSize',14)
set(ax,'XGrid','on', 'YGrid','on', 'Box', 'on');
set(ax,'YColor',[0,0,0]);
set(ax,'YLim', [0 0.5])
set(ax,'XLim', [0 160])
set(ax,'Color','w')
l = legend('test data','Arruda-Boyce', 'Ogden','Neo-Hooke', 'Mooney-Rivlin' ,'Orientation', 'vertical', 'Location','southeast');

% norm of residuals
norm_arruda = norm(Arruda_Boyce(:,3)- Arruda_Boyce(:,2));
norm_mooney = norm(Mooney_Rivlin(:,3)- Mooney_Rivlin(:,2));
norm_neohook = norm(Neo_Hooke(:,3)- Neo_Hooke(:,2));
norm_ogden = norm(Ogden(:,3)-Ogden(:,2));
%norm_ = norm(Ogden(:,3)-Ogden(:,2));

%% TWO BEST FITTING MODELS

% C10 = 0.8859383074 [MPa];
% C01 = 0.1365494839 [MPa];
% D1_mooney = 1.22250848 [MPa^-1]; ---> D1 = 2/K --> K is the bulk modulus
% Mooney_constants = [C10 C01 D1_mooney]

% C1 = 0.214210367 [MPa];
% alpha = 1.61052759;
% D1_ogden = 1.16707703 [MPa^-1]; ---> D1 = 2/K --> K is the bulk modulus				
% Ogden_constants = [C1 alpha D1_ogden]

figure
plot(Yeoh(:,1)*100, Yeoh(:,2), 'or')
title("\textbf{Linear elastic part (approximation)}",'Interpreter','latex','FontSize',30)
ylabel('\textbf{Stress [MPa]}','Interpreter','latex','FontSize',25)
xlabel('\textbf{Strain [$\%$]}','Interpreter','latex','FontSize',25)
ax = gca;
set(ax,'FontSize',14)
set(ax,'XGrid','on', 'YGrid','on', 'Box', 'on');
set(ax,'YColor',[0,0,0]);
set(ax,'YLim', [0 0.5])
set(ax,'XLim', [0 10])
set(ax,'Color','w')


E = (0.06988 - 0)/(0.1-0); % 0.6988 [MPa] 