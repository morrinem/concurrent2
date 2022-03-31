# concurrent2

## connecting to stoker
connect using openVPN

`ssh -X username@stoker.scss.tcd.ie`

## copying files to stoker
while not ssh'ed

`scp <directory(ies)> username@stoker.scss.tcd.ie:~/<directory>`

## running
`make

`./conv <img width> <img height> <kernel order> <n channels> <n kernels>`

### where
img w/h:		`16-512`
kernel order:	`1,3,5,7`
n channels:		`32-2048 (2^n)`
n kernels:		`32-2048 (2^n)`

## resources used:
https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html

https://arxiv.org/pdf/1704.04428.pdf

https://medium.com/apache-mxnet/multi-channel-convolutions-explained-with-ms-excel-9bbf8eb77108

Lecture slides
