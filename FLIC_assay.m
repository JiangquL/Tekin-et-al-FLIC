%% instructions%%
FilePath = %enter file path here 
MonitorFile= %entire monitor file (.csv) here 
Outputfile = %enter name of the output excel file here
OutPath = %enter path for output file here
Outgraph = %enter name for the graph being stored
% code for helper function is on line 54 onwards
%%
clc
clear all
close all
cd FilePath
 
data = csvread(MonitorFile,1,2);
index = data(:,2)';
W = data(:,3:end);
dt=0.2;
 
S = size(W);
 
for i = 1:S(2)
    t = W(:,i)-median(W(:,i));
    x = index*dt;
     [lat{i},ft{i},lt{i},i2{i},avg_intensity{i},output{i}] = datamanip(t);
     figure
     plot(x,t)
     xlabel("Time(msec)")
     ylabel("AU")
     ylim([-50 500]) 
end
%{
lat=latency(in minutes),ft=feeding time(in seconds),lt=licking time(in
seconds)
%}
filename='Outputfile';
sheet=1;
xlswrite(filename,avg_intensity',sheet);
sheet=2;
xlswrite(filename,lat',sheet);
sheet=3;
xlswrite(filename,lt',sheet);
sheet=4;
xlswrite(filename,ft',sheet);

path='Outpath'; %mention path
myfolder='Outgraph'; %give name to the new folder where the graphs will be stored
folder = mkdir([path,filesep,myfolder]) ;
path  = [path,filesep,myfolder] ;
for k = 1:12
    figure(k);
    temp=[path,filesep,'fig',num2str(k),'.png'];
    saveas(gca,temp);
end
%% code for helper function %%
function [latency_time,feed_time,lick_time,ii2,Avg_Int,Out_mat] = datamanip(t)
 
lick=t(t>=10 & t<=40);
feed=t(t>40);
lick_peaks=size(lick,1);
feed_peaks=size(feed,1);
dt= 0.2; 
 
lick_time= (lick_peaks*dt)
feed_time= (feed_peaks*dt)
 
latency=find(t>=10,1);
latency_time=((latency*dt)/60)
 
idx=t>=20;
trans_idx=idx';
ii1=strfind([0 trans_idx 0],[0 1]);
ii2=strfind([0 trans_idx 0],[1 0])-1;
ii=(ii2-ii1+1)>=5;
 
contact=t(t>10);
val=size(contact,1);
s=sum(contact);
Avg_Int=s/val;
 
 
 
t = t';
out = arrayfun(@(x,y) t(x:y),ii1(ii),ii2(ii),'un',0);
 
if sum(size(out))>1
  
    for i = 1:length(out)
        len(i) = length(cell2mat(out(i)));
    end
    l = max(len);
    r = length(len);
    A = zeros(r,l);
 
    for i = 1:r
        A = cell2mat(out(i));
        Out_mat(i,1:length(A)) = A;
    end
else
    Out_mat = 0;
end
