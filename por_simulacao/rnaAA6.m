function  [sys, x0]  = treinarnaAA6(t,x,u,flag)

if flag == 0

    rand('seed',33);        % Garante repetibilidade do experimento
    randn('seed',55);
    
	ninput=3;				% Num de entradas = x; y; contador
	nout=1;					% Num de saidas = yn; W1(1,5); W2(1,5)
        
	x0 = [0; zeros(5,1); zeros(5,1)]; % Inicializa vetor de estados (coluna)
    
	sys = [0;size(x0,1); nout; ninput;0;0];  % Depois que defini, nunca mudei desde 19xx!

elseif flag == 2
    npop=15;
    Alfa=0.0001;
    PopMax=2*npop;  
    xx=u(1);
    y=u(2);
    contador = u(3);
    Precisao=1e-15;
    
    if contador < 2
        disp('FASE 1 - Criacao da populacao inicial.')
        for i=1:npop %gera a populacao inicial
            individuoW1=0.9*rand(1,5);
            individuoW2=0.9*rand(1,5);
            populacao(i,:)= [individuoW1(1,:) individuoW2(1,:)];               % Introduz novo individuo na populacao
        end
        disp('Populacao inicial criada.')
        save pop_ini populacao
        W1=populacao(1,1:5);
        W2=populacao(1,6:10);
        Yo(1,:)=tansig(W1(1,:)*xx);        % Calcula saida da camada oculta
        Yn=W2(1,:)*Yo(1,:)';              % Calcula saida estimada pela RNA
	
        sys = [Yn; W1'; W2'];
        
    elseif contador >=2 && contador < 30    
        load pop_ini
        disp('FASE 1.1 - Treinando populacao inicial.')
        for i=1:npop %treina a populacao inicial
            individuot=populacao(i,:);
            [individuo,Err]=train(individuot,Alfa,xx,y);
            populacao(i,:)=individuo;
            J(i)=Err;       %custo é o erro (Y-Yn)
        end
        disp('Populacao inicial treinada.')
        [populacao,J]=ordem(populacao,J);   % Coloca na ordem de J
        
        save pop populacao
        save erro J
        W1=populacao(1,1:5);
        W2=populacao(1,6:10);
        Yo(1,:)=tansig(W1(1,:)*xx);        % Calcula saida da camada oculta
        Yn=W2(1,:)*Yo(1,:)';              % Calcula saida estimada pela RNA
	
        sys = [Yn; W1'; W2'];
        
    elseif contador == 30       
        load pop
        load erro
        disp('FASE 2.1 - Aumento da populacao.')
        for i=1:npop
            filho=gera_filho(populacao);% Gera novo filho
            populacao(size(populacao,1)+1,:)=filho;% Insere individuo treinado na populacao
        end
        
        for i=1:PopMax       % Treinamento
            [individuo,Err]=train(populacao(i,:),Alfa,xx,y);
            populacao(i,:)=individuo;
            J(i)=Err;       %custo é o erro (Y-Yn)
        end
        
        [populacao,J]=ordem(populacao,J);   % Coloca na ordem de J
        
        disp('Populacao aumentada.')       
        disp('FASE 2.2 - Seleção natural - Elitismo')
        if size(populacao,1) > npop
            disp('Eliminacao de individuos.');
            populacao=populacao(1:npop,:);     % Ajusta tamanho da populacao (menos 1)
            J=J(1:npop);
        end
        
        save pop populacao
        save erro J
        W1=populacao(1,1:5);
        W2=populacao(1,6:10);
        Yo(1,:)=tansig(W1(1,:)*xx);        % Calcula saida da camada oculta
        Yn=W2(1,:)*Yo(1,:)';              % Calcula saida estimada pela RNA
	
        sys = [Yn; W1'; W2'];
             
    elseif contador > 30 && contador < 50
        load pop
        load erro       
        disp('FASE 2.3 - Aperfeicoamento da populacao evoluida.');
        if contador == 31
            Jfim=1e5;       % So para inicializar variÃ¡veis
            %conta=20;                % Numero maximo de loops de treinamento (evita loop muito longo)
            save Jfim Jfim
            %save conta
        else
            load Jfim
            %load conta
        end
            
        if Jfim>Precisao %&& conta>0    % Forca o treinamento ate alcancar precisao desejada
            for i=1:npop       % Treinamento
                [individuo,Err]=train(populacao(i,:),Alfa,xx,y);
                populacao(i,:)=individuo;
                J(i)=Err;       %custo é o erro (Y-Yn)
            end
            %conta=conta-1;        % Decrementa contador de loops de treinamento
            Jfim=min(J);
            save Jfim Jfim
            %save conta conta
        end
        
        [populacao,J]=ordem(populacao,J);   % Coloca na ordem de J
       
        save pop populacao
        save erro J
        
        W1=populacao(1,1:5);
        W2=populacao(1,6:10);
        Yo(1,:)=tansig(W1(1,:)*xx);        % Calcula saida da camada oculta
        Yn=W2(1,:)*Yo(1,:)';              % Calcula saida estimada pela RNA
        
        sys = [Yn; W1'; W2'];
            
	elseif contador == 50       
        load pop
        load erro
        disp('FASE 2.1 - Aumento da populacao.')
        for i=1:npop
            filho=gera_filho(populacao);% Gera novo filho
            populacao(size(populacao,1)+1,:)=filho;% Insere individuo treinado na populacao
        end
        
        for i=1:PopMax       % Treinamento
            [individuo,Err]=train(populacao(i,:),Alfa,xx,y);
            populacao(i,:)=individuo;
            J(i)=Err;       %custo é o erro (Y-Yn)
        end
        
        [populacao,J]=ordem(populacao,J);   % Coloca na ordem de J
        
        disp('Populacao aumentada.')      
        disp('FASE 2.2 - Seleção natural - Elitismo')
        if size(populacao,1) > npop
            disp('Eliminacao de individuos.');
            populacao=populacao(1:npop,:);     % Ajusta tamanho da populacao (menos 1)
            J=J(1:npop);
        end
        
        save pop populacao
        save erro J
        W1=populacao(1,1:5);
        W2=populacao(1,6:10);
        Yo(1,:)=tansig(W1(1,:)*xx);        % Calcula saida da camada oculta
        Yn=W2(1,:)*Yo(1,:)';              % Calcula saida estimada pela RNA
	
        sys = [Yn; W1'; W2'];
             
    elseif contador > 50 && contador < 60
        load pop
        load erro       
        disp('FASE 2.3 - Aperfeicoamento da populacao evoluida.');
        if contador == 51
            Jfim=1e5;       % So para inicializar variÃ¡veis
            %conta=20;                % Numero maximo de loops de treinamento (evita loop muito longo)
            save Jfim Jfim
            %save conta
        else
            load Jfim
            %load conta
        end
            
        if Jfim>Precisao% && conta>0    % Forca o treinamento ate alcancar precisao desejada
            for i=1:npop       % Treinamento
                [individuo,Err]=train(populacao(i,:),Alfa/10,xx,y);
                populacao(i,:)=individuo;
                J(i)=Err;       %custo é o erro (Y-Yn)
            end
            %conta=conta-1;        % Decrementa contador de loops de treinamento
            Jfim=min(J);
            save Jfim Jfim
            %save conta conta
        end
        
        [populacao,J]=ordem(populacao,J);   % Coloca na ordem de J
       
        save pop populacao
        save erro J
        
        W1=populacao(1,1:5);
        W2=populacao(1,6:10);
        Yo(1,:)=tansig(W1(1,:)*xx);        % Calcula saida da camada oculta
        Yn=W2(1,:)*Yo(1,:)';              % Calcula saida estimada pela RNA
        
        sys = [Yn; W1'; W2'];

            
	elseif contador == 60       
        load pop
        load erro
        disp('FASE 2.1 - Aumento da populacao.')
        for i=1:npop
            filho=gera_filho(populacao);% Gera novo filho
            populacao(size(populacao,1)+1,:)=filho;% Insere individuo treinado na populacao
        end
        
        for i=1:PopMax       % Treinamento
            [individuo,Err]=train(populacao(i,:),Alfa/10,xx,y);
            populacao(i,:)=individuo;
            J(i)=Err;       %custo é o erro (Y-Yn)
        end
        
        [populacao,J]=ordem(populacao,J);   % Coloca na ordem de J
        
        disp('Populacao aumentada.')
        disp('FASE 2.2 - Seleção natural - Elitismo')
        if size(populacao,1) > npop
            disp('Eliminacao de individuos.');
            populacao=populacao(1:npop,:);     % Ajusta tamanho da populacao (menos 1)
            J=J(1:npop);
        end
        
        save pop populacao
        save erro J
        W1=populacao(1,1:5);
        W2=populacao(1,6:10);
        Yo(1,:)=tansig(W1(1,:)*xx);        % Calcula saida da camada oculta
        Yn=W2(1,:)*Yo(1,:)';              % Calcula saida estimada pela RNA
	
        sys = [Yn; W1'; W2'];
             
    elseif contador > 60 && contador < 70
        load pop
        load erro       
        disp('FASE 2.3 - Aperfeicoamento da populacao evoluida.');
        if contador == 61
            Jfim=1e5;       % So para inicializar variÃ¡veis
            %conta=20;                % Numero maximo de loops de treinamento (evita loop muito longo)
            save Jfim Jfim
            %save conta
        else
            load Jfim
            %load conta
        end
            
        if Jfim>Precisao %&& conta>0    % Forca o treinamento ate alcancar precisao desejada
            for i=1:npop       % Treinamento
                [individuo,Err]=train(populacao(i,:),Alfa/10,xx,y);
                populacao(i,:)=individuo;
                J(i)=Err;       %custo é o erro (Y-Yn)
            end
            %conta=conta-1;        % Decrementa contador de loops de treinamento
            Jfim=min(J);
            save Jfim Jfim
            %save conta conta
        end
        
        [populacao,J]=ordem(populacao,J);   % Coloca na ordem de J
       
        save pop populacao
        save erro J
        
        W1=populacao(1,1:5);
        W2=populacao(1,6:10);
        Yo(1,:)=tansig(W1(1,:)*xx);        % Calcula saida da camada oculta
        Yn=W2(1,:)*Yo(1,:)';              % Calcula saida estimada pela RNA
        
        sys = [Yn; W1'; W2'];
        
	elseif contador == 70       
        load pop
        load erro
        disp('FASE 2.1 - Aumento da populacao.')
        for i=1:npop
            filho=gera_filho(populacao);% Gera novo filho
            populacao(size(populacao,1)+1,:)=filho;% Insere individuo treinado na populacao
        end
        
        for i=1:PopMax       % Treinamento
            [individuo,Err]=train(populacao(i,:),Alfa/10,xx,y);
            populacao(i,:)=individuo;
            J(i)=Err;       %custo é o erro (Y-Yn)
        end
        
        [populacao,J]=ordem(populacao,J);   % Coloca na ordem de J
        
        disp('Populacao aumentada.')
        
        disp('FASE 2.2 - Seleção natural - Elitismo')
        if size(populacao,1) > npop
            disp('Eliminacao de individuos.');
            populacao=populacao(1:npop,:);     % Ajusta tamanho da populacao (menos 1)
            J=J(1:npop);
        end
        
        save pop populacao
        save erro J
        W1=populacao(1,1:5);
        W2=populacao(1,6:10);
        Yo(1,:)=tansig(W1(1,:)*xx);        % Calcula saida da camada oculta
        Yn=W2(1,:)*Yo(1,:)';              % Calcula saida estimada pela RNA
	
        sys = [Yn; W1'; W2'];
             
    elseif contador > 70 && contador < 99
        load pop
        load erro       
        disp('FASE 3 - Aperfeicoamento da populacao final.');
        if contador == 71
            Jfim=1e5;       % So para inicializar variÃ¡veis
            save Jfim Jfim
        else
            load Jfim
        end
            
        if Jfim>Precisao
            for i=1:npop       % Treinamento
                [individuo,Err]=train(populacao(i,:),Alfa/1000,xx,y);
                populacao(i,:)=individuo;
                J(i)=Err;       %custo é o erro (Y-Yn)
            end          
            Jfim=min(J);
            save Jfim Jfim
        end
        
        [populacao,J]=ordem(populacao,J);   % Coloca na ordem de J
       
        save pop populacao
        save erro J
        
        W1=populacao(1,1:5);
        W2=populacao(1,6:10);
        Yo(1,:)=tansig(W1(1,:)*xx);        % Calcula saida da camada oculta
        Yn=W2(1,:)*Yo(1,:)';              % Calcula saida estimada pela RNA
        
        sys = [Yn; W1'; W2'];
        
    elseif contador >= 99
        load pop
        load erro
        
        [populacao,J]=ordem(populacao,J);   % Coloca na ordem de J
        
        disp('Populacao final aperfeicoada.')
        save pop populacao
        save erro J
        W1=populacao(1,1:5);
        W2=populacao(1,6:10);
        Yo(1,:)=tansig(W1(1,:)*xx);        % Calcula saida da camada oculta
        Yn=W2(1,:)*Yo(1,:)';              % Calcula saida estimada pela RNA
        
        errofim = J(1);
        
        disp('O melhor individuo da populacao final eh o escolhido.')
        disp('Este individuo eh composto pelos pesos:')
        disp('W1:')
        disp(W1)
        disp('e W2:')
        disp(W2)
        disp('O respectivo erro deste individuo eh:')
        disp(errofim)
        
        save pesosW1 W1
        save pesosW2 W2
        save errofim errofim
	
        sys = [Yn; W1'; W2'];
        
    end
  
elseif flag == 3
    sys = x(1);
end
end

%==========================================================================
%FUNCOES DO PROGRAMA
%==========================================================================
function [nind,J]=train(individuo,passo,xx,y);

funcao="y - (W2(1,:)*(tansig(W1(1,:)*xx)'))"; % e = Y(i)-Yn(i)
grad='[-W2(1,:).*(1-(W1(1,:)*xx).^2)*xx      -tansig(W1(1,:)*xx)]';

% Treinamento de um individuo
W1=individuo(1,1:5);
W2=individuo(1,6:10);

V1=abs(eval(funcao));           % Guarda valor da funcao no ponto inicial

hk=eval(grad);         % Direcao do gradiente negativo
lambda=calc_min(funcao,hk,W1,W2,passo,xx,y);   % Calcula o melhor passo nesta direcao
%lambda = [passo passo]; 
W1=W1-lambda(1)*hk(1,1:5);          % Atualiza xk na direcao hk com o melhor passo possivel
W2=W2-lambda(2)*hk(1,6:10);
V=abs(eval(funcao));         % Novo valor da funcao

if V1 > V             % Funcao custo aumentou de valor
    disp('Custo diminuindo, treinamento convergindo');
else
	disp('Custo aumentando, risco de nao convergir');
end
J=V;                      % Custo alcancado com o treinamento   
nind=[W1 W2];             % Atualiza novo individuo
end
%==========================================================================
function lambda=calc_min(funcao,hk,W1k,W2k,passo,xx,y);
% Calculo de minimo unidimensional utilizando metodo de busca uniforme
tol=1e-6;             % Tolerancia para condicao de saida do loop de busca
if passo<tol          % So para evitar inconsistencias
   tol=passo/100;
end
W1=W1k;                 % Inicializa x de busca
W2=W2k;
parou=0;              % Verifica se algoritmo nao consegue a tolerancia desejada
while abs(passo)>tol & ~parou
   Jk=abs(eval(funcao));   % Valor da funcao no ponto x anterior
   W1=W1+passo*hk(1,1:5);      % Adianta 1 passo
   W2=W2+passo*hk(1,6:10);
   Jk1=abs(eval(funcao));  % Valor da funcao no novo ponto x
   delta= Jk1-Jk;     % Variacao no valor da funcao custo
   if delta > 0       % Funcao no sentido crescente, esta indo para o lado errado
      passo=-passo/2; % Inverte o sentido e diminui o passo
   end
   if delta<tol
      parou=1;
   end
end
lambda1=(W1-W1k)/hk(1,1:5);     % Calcula o valor de lambda para o ponto: x = xk + lambda*hk
lambda2=(W2-W2k)/hk(1,6:10);
lambda=[lambda1 lambda2];
end
%==========================================================================
function filho=gera_filho(populacao); % Gera novo filho
TamP=size(populacao,1);              % Tamanho da populacao (PAIS e MAES possiveis)
BestP=ceil(TamP*0.2);                % Garante um pai dentro dos 20% melhores 
nPai=ceil(rand*BestP);               % Seleciona pai dentro dos BestP melhores
nMae=ceil(rand*TamP)                              % So para inicializar variaveis
if nMae==nPai               % Individuo pai diferente do individuo mae
    nMae=ceil(rand*TamP);       % Seleciona mae dentro de toda a populacao
end
pai=populacao(nPai,:);            % Dados do Pai na populacao
mae=populacao(nMae,:);            % Dados da Mae na populacao
filho=mean([pai;mae])+0.9*rand(size(pai))*sqrt(sumsqr(pai-mae))/2;  % Euristica para gerar o filho
% Probabilidade de gerar filho com super-mutacao 
if rand<0.30                      % Probabilidade de 30% de gerar super-mutante
    Fator=max(sqrt([sumsqr(pai) sumsqr(mae)  1])); % Maior distancia euclidiana (>=1)
    filho=0.9*rand*(pai-mae);         % Direcao e sentido para Super-Mutacao
    filho=Fator*filho+mean([pai;mae]); % Novo ponto de busca (distante)
end
end
%==========================================================================
function [npopulacao,nJ]=ordem(populacao,J);   % Coloca na ordem de J
[Y,I]=sort(J);          % Coloca em ordem de J e sai com indice I
for i=1:length(I)       % Varre o indice
   npopulacao(i,:)=populacao(I(i),:);
   nJ(i)=J(I(i));
end
end
%==========================================================================