main_dir = '/Users/yikai/Desktop/imagenet-dataset/'

% fileFolder=fullfile('/Users/yikai/Desktop/imagenet-dataset/source/');

dirOutput=dir(fullfile(main_dir,'source/', '*.bin'));
 
fileNames={dirOutput.name};
flen=length(fileNames);

A={};
B={};
for i=1:flen
    A=[A,'A'];
    B=[B,'B'];
end
A_temp=genvarname(A);
B_temp=genvarname(B);

for i =1:flen 
    temp=strcat(main_dir,'source/', fileNames(i));
    fna=fileNames(i);
    na=strsplit(fna{1},'.');
    na1=na(1);
    na1=na1{1};
    temp=temp{1};
    disp(temp)
    X=loads(temp);
    path1=[main_dir, 'feature/', na1,'.mat'];
    path2=[main_dir, 'label/', na1,'.mat'];
    res=str2num(na1);
    resm=[];
    resm(1)=res;
    disp(path1);
    save(path1,'X');
    disp(path2);
    save(path2,'resm');
    
    nn1=strcat('X' ,na1);
    nn2=strcat('resm' ,na1);

    eval( [nn1, '= X']);
    eval( [nn2, '= resm']);

    if i==1
        save('/Users/yikai/Desktop/ffeature.mat',['X',na1]);  
        save('/Users/yikai/Desktop/flabel.mat',['resm',na1]);
    else
        save('/Users/yikai/Desktop/ffeature.mat',['X',na1],'-append')  
        save('/Users/yikai/Desktop/flabel.mat',['resm',na1],'-append')  
    end
   
end

