import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio

from functools import reduce
from matplotlib import cm



class ImageGrid(object):

    def __init__(self, data, res_h=128, res_w=128):
        self.data = data
        if len(self.data.shape) < 3:
            self.data = np.tile(self.data[...,np.newaxis], [1,1,3])
        self.res_h = res_h
        self.res_w = res_w
        self.w = self.data.shape[1] // self.res_w
        self.h = self.data.shape[0] // self.res_h

    @property
    def shape(self):
        return (self.h, self.w)

    @property
    def blockShape(self):
        return (self.res_h, self.res_w)


    def _reshape(self, new_h, new_w):
        if new_h == -1:
            if (self.w*self.h) % new_h !=0:
                raise ValueError()
            new_h = self.w*self.h // new_w

        if new_w == -1:
            if (self.w*self.h) % new_w !=0:
                raise ValueError()
            new_w = self.w*self.h // new_h



        if new_h*new_w != self.w*self.h:
            raise ValueError("Cannot reshape shape {}/{} into {}/{}".format(self.h, self.w, new_h, new_w))


        rows = np.split(self.data, self.h, axis=0)
        rows = np.array(rows)
        blocks = np.split(rows, self.w, axis=2)
        blocks = np.array(blocks)
        blocks = np.transpose(blocks, [1,0,4,2,3])
        blocks = np.reshape(blocks, [new_h, new_w]+list(blocks.shape[2:]))
        return blocks


    def resize(self, new_h_res, new_w_res, method=Image.LANCZOS):
        img = Image.fromarray(self.data)
        img = img.resize((new_w_res, new_h_res), method)
        img = np.asarray(img)

        return ImageGrid(img, new_h_res//self.shape[0], new_w_res//self.shape[1])


    def resize_by_scale(self, scale, method=Image.LANCZOS):
        new_h = self.data.shape[0]*scale
        new_w = self.data.shape[1]*scale
        return self.resize(new_h, new_w, method=method)

    def reshape(self, new_h, new_w):

        blocks = self._reshape(new_h, new_w)
        blocks = list(blocks)
        blocks = list(map(lambda x: list(x), blocks))
        blocks = np.block(blocks)
        blocks = np.transpose(blocks, [1,2,0])

        return ImageGrid(blocks, res_h=self.res_h, res_w=self.res_w)

    def transpose(self):
        blocks = self._reshape(self.h, self.w)
        blocks = np.transpose(blocks, [1, 0, 2, 3, 4])
        blocks = list(blocks)
        blocks = list(map(lambda x: list(x), blocks))
        blocks = np.block(blocks)
        blocks = np.transpose(blocks, [1,2,0])

        return ImageGrid(blocks, res_h=self.res_h, res_w=self.res_w)

    def pad(self, h_pad, w_pad, value=(255,255,255)):
        value = np.array(value)[...,np.newaxis,np.newaxis]
        temp = self._reshape(self.h, self.w)
        temp = np.pad(temp, [(0,0), (0,0), (0,0), (h_pad,h_pad), (w_pad,w_pad)], mode="constant", constant_values=255)
        if h_pad > 0:
            temp[...,:h_pad,:] = value
            temp[...,-h_pad:,:] = value
        if w_pad > 0:
            temp[...,:,:w_pad] = value
            temp[...,:,-w_pad:] = value
        blocks = list(temp)
        blocks = list(map(lambda x: list(x), blocks))
        blocks = np.block(blocks)
        blocks = np.transpose(blocks, [1,2,0])


        return ImageGrid(blocks, res_h=self.res_h+2*h_pad, res_w=self.res_w+2*w_pad)


    def split(self, axis, sp=None):
        if sp is None:
            sp = self.h if axis==0 else self.w
        out = np.split(self.data, sp, axis)

        return list(map(lambda x: ImageGrid(x, res_h=self.res_h, res_w=self.res_w), out))


    def caption_frame(self, in_str, font_size=22, adapt_color=False, color=None, bottom=False):
        sh = self.data.shape
        img = Image.fromarray(self.data)
        if np.mean(self.data[0:75, 0:40, :]) > 200 and adapt_color:    
            text_col = (0,0,0)
        else:
            text_col = (255,255,255)

        if color is not None:
            text_col = color
        font = ImageFont.truetype("LiberationSerif-Regular.ttf", font_size)
        draw = ImageDraw.Draw(img)
        if bottom:
            draw.text((0,self.res_h-font_size), in_str, text_col, font=font)
        else:
            draw.text((0,0), in_str, text_col, font=font)
        img = np.asarray(img)
        out = ImageGrid(img, res_h=sh[0], res_w=sh[1])
        return out


    def caption_flat(self, in_str_list, font_size=32, color=(255,0,0), bottom=False):

        sh = self.shape
        t = self.reshape(1, -1)
        t = t.split(1)


        N = len(t)
        if type(in_str_list) is not list:
            in_str_list = [in_str_list for _ in range(N)]

        t = list(map(lambda t: t[0].caption_frame(t[1], font_size=font_size, color=color, bottom=bottom), zip(t, in_str_list)))
        t = ImageGrid.stackV(t)

        return t.reshape(sh[0], sh[1])


    def group(self):
        return ImageGrid(self.data, res_h=self.data.shape[0], res_w=self.data.shape[1])

    def group_horizontal(self):
        return ImageGrid(self.data, res_h=self.res_h, res_w=self.data.shape[1])

    def group_vertical(self):
        return ImageGrid(self.data, res_h=self.data.shape[0], res_w=self.res_w)



    def extract_cols(self, cols):
        temp = []
        for col in cols:
            temp += list(range(col*self.res_w, (col+1)*self.res_w))
        temp = self.data[:, temp, :]
        return ImageGrid(temp, res_h=self.res_h, res_w=self.res_w)

    def extract_rows(self, rows):
        temp = []
        for row in rows:
            temp += list(range(row*self.res_h, (row+1)*self.res_h))
        temp = self.data[temp, :, :]
        return ImageGrid(temp, res_h=self.res_h, res_w=self.res_w)


    def __add__(self, other):
        if self.res_h != other.res_h:
            raise ValueError("Cannot concatenate grids with different horizontal block resolutions, was {}/{}".format(self.res_h, other.res_h))
        temp = np.concatenate([self.data, other.data], axis=1)
        return ImageGrid(temp.astype(np.uint8), res_h=self.res_h, res_w=self.res_w)

    def __mul__(self, other):
        if self.res_w != other.res_w:
            raise ValueError("Cannot concatenate grids with different vertical block resolutions, was {}/{}".format(self.res_w, other.res_w))
        temp = np.concatenate([self.data, other.data], axis=0)
        return ImageGrid(temp.astype(np.uint8), res_h=self.res_h, res_w=self.res_w)



    def mirror(self, axis):
        cols = np.array(list(map(lambda x: x.data, np.array(self.split(axis)))))
        cols = np.flip(cols, axis=0)
        cols = list(map(lambda x: ImageGrid(x, res_h=self.res_h, res_w=self.res_w), cols))

        if axis==0:
            return ImageGrid.stackH(cols)
        else:
            return ImageGrid.stackV(cols)


    def toVideo(self, reshape_y_axis, fps=24, mirror=False, filename="out.mp4", modifying_class_func=lambda *x: x[0], modifying_class_func_args=()):
        frames = self.split(1)
        frames = list(map(lambda x: modifying_class_func(x.reshape(*reshape_y_axis), *modifying_class_func_args).data.astype(np.uint8), frames))
        frames = np.array(frames).astype(np.uint8)

        if mirror:
            frames = np.concatenate([frames, np.flip(frames, axis=0)], axis=0)

        writer = imageio.get_writer(filename, mode="I", codec="libx264", bitrate="6M", fps=fps, output_params=["-sws_flags","neighbor","-sws_dither", "none"])  
        for frame in frames:
            writer.append_data(frame)
        writer.close()

    def interleave(self, others, axis, interval):
        vals = [self.split(axis, sp=interval)] + list(map(lambda x: x.split(axis, sp=interval), others))
        out = None

        for all_vals in zip(*vals):
            if axis==0:
                ab = reduce(lambda x,y: x*y, all_vals)
                out = out * ab if out is not None else ab
            else:
                ab = reduce(lambda x,y: x+y, all_vals)
                out = out + ab if out is not None else ab

        return out



    @classmethod
    def from_bw(cls, res, data, normalize=False, colormap="viridis"):
        colors = cm.get_cmap(colormap)
        if normalize:
            norm_data = (data-np.min(data))/(np.max(data)-np.min(data))
        else:
            norm_data = data
        if colormap=="Greys":
            norm_data = 1.0 - norm_data
        color_data = colors(norm_data)
        color_data = color_data.reshape(-1,res,4)[:,:,0:3]*255.0
        color_data = color_data.astype(np.uint8)

        out = cls(color_data, res_w=res, res_h=res)
        return out



    @staticmethod
    def stackV(ll):
        return reduce(lambda x,y: x+y, ll)

    @staticmethod
    def stackH(ll):
        return reduce(lambda x,y: x*y, ll)

    @classmethod
    def fromFile(cls, path, res):
        dat = Image.open(path)
        dat = np.asarray(dat)
        return cls(dat, res_h=res, res_w=res)


    def save(self, name="out.png", bw=False, quality=100):
        img = Image.fromarray(self.data)
        if bw:
            img = img.convert("L")
        img.save(name, quality=quality)


    def save_individual_imgs(self, name="out", quality=100, individual_save_labels=[], use_coordinates_as_labels=False):
        return self.save_individual_images(name=name, quality=quality, individual_save_labels=individual_save_labels, use_coordinates_as_labels=use_coordinates_as_labels)

    def save_individual_images(self, name="out", quality=100, individual_save_labels=[], use_coordinates_as_labels=False):
        _,W = self.shape
        imgs = self.reshape(1,-1).split(axis=1)
        N = len(imgs)
        if individual_save_labels:
            assert N == len(individual_save_labels)
            for i,lab in enumerate(individual_save_labels):
                img = imgs[i]
                img = Image.fromarray(img.data)
                img.save("{}_{}.png".format(name, lab), quality=quality)
        else:
            for i in range(N):
                if use_coordinates_as_labels:
                    row = i//W
                    col = i%W
                    img = imgs[i]
                    img = Image.fromarray(img.data)
                    img.save("{}_R{}_C{}.png".format(name, str(row).zfill(2), str(col).zfill(2)), quality=quality)
                else:
                    img = imgs[i]
                    img = Image.fromarray(img.data)
                    img.save("{}_{}.png".format(name, i), quality=quality)