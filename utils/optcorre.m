function [ pvalue, corre] = optcorre( img1, img2 )

row=size(img1,1);
col=size(img1,2);
fresult=fft2(img1);
fresult=fftshift(fresult);
%fresult=fft2(Adouble);

kk=0.3;
fadouble=fft2(img2);
fadouble=fftshift(fadouble);

corre=ifft2(((abs(fresult).*abs(fadouble)).^kk).*exp(1i*(angle(fresult)-angle(fadouble))));
corre=abs(fftshift(corre));

peak=max(max(corre));
avg=mean(mean(corre));
pvalue=peak/avg

end

