function [idx_patch,idx_patch_t]=batch_img(str1,str2,idx_patch,idx_patch_t,num,scale,patch_wid,patch_hei,Fl)
norubbish_data1=readallimage(str1,'*.bmp',0);
norubbish_data2=readallimage(str2,'*.bmp',0);
load('Cal.mat')
load('Mach.mat')
eval(['stereoParams=stereoParams_',num2str(num),';']);
eval(['Mach=Mach_',num2str(num),';']);
Fl=Fl(num);
for kk=1:1
    if size(Mach,1)~=1
        I1=squeeze(norubbish_data1(:,:,kk));
        I2=squeeze(norubbish_data2(:,:,kk));
        [m,n]=size(I1);
        delta=fix(size(Mach,1)/2000);
        if delta==0
            delta=1;
        end
        for ii=1:delta:size(Mach,1)
            y_pix_L=round(Mach(ii,1));
            x_pix_L=round(Mach(ii,2));
            y_pix_R=round(Mach(ii,3));
            x_pix_R=round(Mach(ii,4));
            if (x_pix_L<=patch_wid)||(x_pix_L>(m-patch_wid))
                continue;
            end
            if (x_pix_R<=patch_wid)||(x_pix_R>(m-patch_wid))
                continue;
            end
            if (y_pix_L<=patch_hei)||(y_pix_L>(n-patch_hei))
                continue;
            end
            if (y_pix_R<=patch_hei)||(y_pix_R>(n-patch_hei))
                continue;
            end
            I1_temp=I1((x_pix_L-(patch_wid)/2+1):(x_pix_L+(patch_wid)/2),(y_pix_L-(patch_hei)/2+1):(y_pix_L+(patch_hei)/2));
            I2_temp=I2((x_pix_R-(patch_wid)/2+1):(x_pix_R+(patch_wid)/2),(y_pix_R-(patch_hei)/2+1):(y_pix_R+(patch_hei)/2));
            img_hr_0 = I1_temp;
            img_hr_1 = I2_temp;
            img_lr_0 = imresize(img_hr_0, 1/scale, 'bicubic');
            img_lr_1 = imresize(img_hr_1, 1/scale, 'bicubic');
            hr_patch_0 = img_hr_0;
            hr_patch_1 = img_hr_1;
            lr_patch_0 = img_lr_0;
            lr_patch_1 = img_lr_1;
            hr_patch_0=repmat(hr_patch_0,1,1,3);
            hr_patch_1=repmat(hr_patch_1,1,1,3);
            lr_patch_0=repmat(lr_patch_0,1,1,3);
            lr_patch_1=repmat(lr_patch_1,1,1,3);
            if Fl
                mkdir(['./train_n/patches_x', num2str(scale), '/', num2str(idx_patch, '%06d')]);
                imwrite(hr_patch_0, ['./train_n/patches_x', num2str(scale), '/', num2str(idx_patch, '%06d'), '/hr0.png']);
                imwrite(hr_patch_1, ['./train_n/patches_x', num2str(scale), '/', num2str(idx_patch, '%06d'), '/hr1.png']);
                imwrite(lr_patch_0, ['./train_n/patches_x', num2str(scale), '/', num2str(idx_patch, '%06d'), '/lr0.png']);
                imwrite(lr_patch_1, ['./train_n/patches_x', num2str(scale), '/', num2str(idx_patch, '%06d'), '/lr1.png']);
                if(mod(idx_patch,10)==0)
                    fprintf([num2str(idx_patch, '%06d'), ' training samples have been generated...\n']);
                end
                idx_patch = idx_patch + 1;
                mkdir(['./test_n/patches_x', num2str(scale), '/hr/', num2str(idx_patch_t, '%06d')]);
                mkdir(['./test_n/patches_x', num2str(scale), '/lr/', num2str(idx_patch_t, '%06d')]);
                imwrite(hr_patch_0, ['./test_n/patches_x', num2str(scale), '/hr/', num2str(idx_patch_t, '%06d'), '/hr0.png']);
                imwrite(hr_patch_1, ['./test_n/patches_x', num2str(scale), '/hr/', num2str(idx_patch_t, '%06d'), '/hr1.png']);
                imwrite(lr_patch_0, ['./test_n/patches_x', num2str(scale), '/lr/', num2str(idx_patch_t, '%06d'), '/lr0.png']);
                imwrite(lr_patch_1, ['./test_n/patches_x', num2str(scale), '/lr/', num2str(idx_patch_t, '%06d'), '/lr1.png']);
                if(mod(idx_patch_t,10)==0)
                    fprintf([num2str(idx_patch_t, '%06d'), ' testing samples have been generated...\n']);
                end
                idx_patch_t = idx_patch_t + 1;
            else
                mkdir(['./test_n/patches_x', num2str(scale), '/hr/', num2str(idx_patch_t, '%06d')]);
                mkdir(['./test_n/patches_x', num2str(scale), '/lr/', num2str(idx_patch_t, '%06d')]);
                imwrite(hr_patch_0, ['./test_n/patches_x', num2str(scale), '/hr/', num2str(idx_patch_t, '%06d'), '/hr0.png']);
                imwrite(hr_patch_1, ['./test_n/patches_x', num2str(scale), '/hr/', num2str(idx_patch_t, '%06d'), '/hr1.png']);
                imwrite(lr_patch_0, ['./test_n/patches_x', num2str(scale), '/lr/', num2str(idx_patch_t, '%06d'), '/lr0.png']);
                imwrite(lr_patch_1, ['./test_n/patches_x', num2str(scale), '/lr/', num2str(idx_patch_t, '%06d'), '/lr1.png']);
                if(mod(idx_patch_t,10)==0)
                    fprintf([num2str(idx_patch_t, '%06d'), ' testing samples have been generated...\n']);
                end
                idx_patch_t = idx_patch_t + 1;
            end
        end
    end
end


