% 
% %%
% %non_ep
% clc
% clear
% idx_patch = 1;
% idx_patch_t = 1;
% scale=2;
% patch_wid=120;
% patch_hei=120;
% p = randperm(22);
% Fl=zeros(1,22);
% Fl(p(1:18))=1;
% str1='C:\Users\Yzy_seu\Desktop\Data\01\Img\U3-328xCP-M(4103787700)\';
% str2='C:\Users\Yzy_seu\Desktop\Data\01\Img\U3-328xCP-M(4103813611)\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,1,scale,patch_wid,patch_hei,Fl);
% str1='C:\Users\Yzy_seu\Desktop\Data\02\Img\U3-328xCP-M(4103787700)\';
% str2='C:\Users\Yzy_seu\Desktop\Data\02\Img\U3-328xCP-M(4103813611)\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,2,scale,patch_wid,patch_hei,Fl);
% str1='C:\Users\Yzy_seu\Desktop\Data\03\Img\0\';
% str2='C:\Users\Yzy_seu\Desktop\Data\03\Img\1\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,3,scale,patch_wid,patch_hei,Fl);
% str1='C:\Users\Yzy_seu\Desktop\Data\04\Img\U3-328xCP-M(4103787700)\';
% str2='C:\Users\Yzy_seu\Desktop\Data\04\Img\U3-328xCP-M(4103813611)\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,4,scale,patch_wid,patch_hei,Fl);
% str1='C:\Users\Yzy_seu\Desktop\Data\05\Img\Camera0\';
% str2='C:\Users\Yzy_seu\Desktop\Data\05\Img\Camera1\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,5,scale,patch_wid,patch_hei,Fl);
% str1='C:\Users\Yzy_seu\Desktop\Data\06\Img\Camera0\';
% str2='C:\Users\Yzy_seu\Desktop\Data\06\Img\Camera1\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,6,scale,patch_wid,patch_hei,Fl);
% str1='C:\Users\Yzy_seu\Desktop\Data\07\Img\0\';
% str2='C:\Users\Yzy_seu\Desktop\Data\07\Img\1\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,7,scale,patch_wid,patch_hei,Fl);
% str1='C:\Users\Yzy_seu\Desktop\Data\08\Img\0\';
% str2='C:\Users\Yzy_seu\Desktop\Data\08\Img\1\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,8,scale,patch_wid,patch_hei,Fl);
% str1='C:\Users\Yzy_seu\Desktop\Data\09\Img\Camera0\';
% str2='C:\Users\Yzy_seu\Desktop\Data\09\Img\Camera1\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,9,scale,patch_wid,patch_hei,Fl);
% str1='C:\Users\Yzy_seu\Desktop\Data\10\Img\Camera0\';
% str2='C:\Users\Yzy_seu\Desktop\Data\10\Img\Camera1\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,10,scale,patch_wid,patch_hei,Fl);
% str1='C:\Users\Yzy_seu\Desktop\Data\11\Img\Camera0\';
% str2='C:\Users\Yzy_seu\Desktop\Data\11\Img\Camera1\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,11,scale,patch_wid,patch_hei,Fl);
% str1='C:\Users\Yzy_seu\Desktop\Data\12\Img\Camera0\';
% str2='C:\Users\Yzy_seu\Desktop\Data\12\Img\Camera1\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,12,scale,patch_wid,patch_hei,Fl);
% str1='C:\Users\Yzy_seu\Desktop\Data\13\Img\Camera0\';
% str2='C:\Users\Yzy_seu\Desktop\Data\13\Img\Camera1\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,13,scale,patch_wid,patch_hei,Fl);
% str1='C:\Users\Yzy_seu\Desktop\Data\14\Img\Camera0\';
% str2='C:\Users\Yzy_seu\Desktop\Data\14\Img\Camera1\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,14,scale,patch_wid,patch_hei,Fl);
% str1='C:\Users\Yzy_seu\Desktop\Data\15\Img\Camera0\';
% str2='C:\Users\Yzy_seu\Desktop\Data\15\Img\Camera1\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,15,scale,patch_wid,patch_hei,Fl);
% str1='C:\Users\Yzy_seu\Desktop\Data\16\Img\0\';
% str2='C:\Users\Yzy_seu\Desktop\Data\16\Img\1\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,16,scale,patch_wid,patch_hei,Fl);
% str1='C:\Users\Yzy_seu\Desktop\Data\17\Img\0\';
% str2='C:\Users\Yzy_seu\Desktop\Data\17\Img\1\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,17,scale,patch_wid,patch_hei,Fl);
% str1='C:\Users\Yzy_seu\Desktop\Data\18\Img\Camera0\';
% str2='C:\Users\Yzy_seu\Desktop\Data\18\Img\Camera1\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,18,scale,patch_wid,patch_hei,Fl);
% str1='C:\Users\Yzy_seu\Desktop\Data\19\Img\Camera0\';
% str2='C:\Users\Yzy_seu\Desktop\Data\19\Img\Camera1\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,19,scale,patch_wid,patch_hei,Fl);
% str1='C:\Users\Yzy_seu\Desktop\Data\20\Img\Camera0\';
% str2='C:\Users\Yzy_seu\Desktop\Data\20\Img\Camera1\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,20,scale,patch_wid,patch_hei,Fl);
% str1='C:\Users\Yzy_seu\Desktop\Data\21\Img\0\';
% str2='C:\Users\Yzy_seu\Desktop\Data\21\Img\1\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,21,scale,patch_wid,patch_hei,Fl);
% str1='C:\Users\Yzy_seu\Desktop\Data\22\Img\Camera0\';
% str2='C:\Users\Yzy_seu\Desktop\Data\22\Img\Camera1\';
% [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,22,scale,patch_wid,patch_hei,Fl);
%%
%after_ep
clc
clear
idx_patch = 1;
idx_patch_t = 1;
scale=2;
patch_wid=120;
patch_hei=120;
p = randperm(22);
Fl=zeros(1,22);
Fl(p(1:18))=1;
str1='C:\Users\Yzy_seu\Desktop\Data\01\Img\U3-328xCP-M(4103787700)\';
str2='C:\Users\Yzy_seu\Desktop\Data\01\Img\U3-328xCP-M(4103813611)\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,1,scale,patch_wid,patch_hei,Fl);
str1='C:\Users\Yzy_seu\Desktop\Data\02\Img\U3-328xCP-M(4103787700)\';
str2='C:\Users\Yzy_seu\Desktop\Data\02\Img\U3-328xCP-M(4103813611)\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,2,scale,patch_wid,patch_hei,Fl);
str1='C:\Users\Yzy_seu\Desktop\Data\03\Img\0\';
str2='C:\Users\Yzy_seu\Desktop\Data\03\Img\1\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,3,scale,patch_wid,patch_hei,Fl);
str1='C:\Users\Yzy_seu\Desktop\Data\04\Img\U3-328xCP-M(4103787700)\';
str2='C:\Users\Yzy_seu\Desktop\Data\04\Img\U3-328xCP-M(4103813611)\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,4,scale,patch_wid,patch_hei,Fl);
str1='C:\Users\Yzy_seu\Desktop\Data\05\Img\Camera0\';
str2='C:\Users\Yzy_seu\Desktop\Data\05\Img\Camera1\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,5,scale,patch_wid,patch_hei,Fl);
str1='C:\Users\Yzy_seu\Desktop\Data\06\Img\Camera0\';
str2='C:\Users\Yzy_seu\Desktop\Data\06\Img\Camera1\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,6,scale,patch_wid,patch_hei,Fl);
str1='C:\Users\Yzy_seu\Desktop\Data\07\Img\0\';
str2='C:\Users\Yzy_seu\Desktop\Data\07\Img\1\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,7,scale,patch_wid,patch_hei,Fl);
str1='C:\Users\Yzy_seu\Desktop\Data\08\Img\0\';
str2='C:\Users\Yzy_seu\Desktop\Data\08\Img\1\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,8,scale,patch_wid,patch_hei,Fl);
str1='C:\Users\Yzy_seu\Desktop\Data\09\Img\Camera0\';
str2='C:\Users\Yzy_seu\Desktop\Data\09\Img\Camera1\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,9,scale,patch_wid,patch_hei,Fl);
str1='C:\Users\Yzy_seu\Desktop\Data\10\Img\Camera0\';
str2='C:\Users\Yzy_seu\Desktop\Data\10\Img\Camera1\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,10,scale,patch_wid,patch_hei,Fl);
str1='C:\Users\Yzy_seu\Desktop\Data\11\Img\Camera0\';
str2='C:\Users\Yzy_seu\Desktop\Data\11\Img\Camera1\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,11,scale,patch_wid,patch_hei,Fl);
str1='C:\Users\Yzy_seu\Desktop\Data\12\Img\Camera0\';
str2='C:\Users\Yzy_seu\Desktop\Data\12\Img\Camera1\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,12,scale,patch_wid,patch_hei,Fl);
str1='C:\Users\Yzy_seu\Desktop\Data\13\Img\Camera0\';
str2='C:\Users\Yzy_seu\Desktop\Data\13\Img\Camera1\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,13,scale,patch_wid,patch_hei,Fl);
str1='C:\Users\Yzy_seu\Desktop\Data\14\Img\Camera0\';
str2='C:\Users\Yzy_seu\Desktop\Data\14\Img\Camera1\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,14,scale,patch_wid,patch_hei,Fl);
str1='C:\Users\Yzy_seu\Desktop\Data\15\Img\Camera0\';
str2='C:\Users\Yzy_seu\Desktop\Data\15\Img\Camera1\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,15,scale,patch_wid,patch_hei,Fl);
str1='C:\Users\Yzy_seu\Desktop\Data\16\Img\0\';
str2='C:\Users\Yzy_seu\Desktop\Data\16\Img\1\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,16,scale,patch_wid,patch_hei,Fl);
str1='C:\Users\Yzy_seu\Desktop\Data\17\Img\0\';
str2='C:\Users\Yzy_seu\Desktop\Data\17\Img\1\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,17,scale,patch_wid,patch_hei,Fl);
str1='C:\Users\Yzy_seu\Desktop\Data\18\Img\Camera0\';
str2='C:\Users\Yzy_seu\Desktop\Data\18\Img\Camera1\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,18,scale,patch_wid,patch_hei,Fl);
str1='C:\Users\Yzy_seu\Desktop\Data\19\Img\Camera0\';
str2='C:\Users\Yzy_seu\Desktop\Data\19\Img\Camera1\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,19,scale,patch_wid,patch_hei,Fl);
str1='C:\Users\Yzy_seu\Desktop\Data\20\Img\Camera0\';
str2='C:\Users\Yzy_seu\Desktop\Data\20\Img\Camera1\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,20,scale,patch_wid,patch_hei,Fl);
str1='C:\Users\Yzy_seu\Desktop\Data\21\Img\0\';
str2='C:\Users\Yzy_seu\Desktop\Data\21\Img\1\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,21,scale,patch_wid,patch_hei,Fl);
str1='C:\Users\Yzy_seu\Desktop\Data\22\Img\Camera0\';
str2='C:\Users\Yzy_seu\Desktop\Data\22\Img\Camera1\';
[idx_patch,idx_patch_t]=batch_img_aftere(str1,str2,idx_patch,idx_patch_t,22,scale,patch_wid,patch_hei,Fl);