$out_dir = 'build';

use Cwd 'abs_path';
$ENV{'TEXINPUTS'} = abs_path('template') . ':';
$ENV{'BSTINPUTS'} = abs_path('template') . ':';
