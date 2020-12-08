

$infile=shift;
$seq_flag=shift;
$outfile=$infile.".coding.onehot";

$seq_flag=$seq_flag+0;
open(out,">$outfile");
open(in,$infile_his);
$line=<in>;
$max_len=-1;
while(defined($line=<in>)){
	chomp($line);
	$line =~ s/\r//g;
	if($line=~/^>/){
		print out $seq_flag,",";
		$tar_seq=<in>;
		$seq_len=length($tar_seq);
		
		for($i=0;$i<$seq_len;$i=$i+1){
			
			$tar_nuc=substr($tar_seq,$i,1);
			if($tar_nuc eq "A"){
				print out "1 ";
				next;
			}
			if($tar_nuc eq "T"){
				print out "2 ";
				next;
			}
			if($tar_nuc eq "G"){
				print out "3 ";
				next;
			}
			if($tar_nuc eq "C"){
				print out "4 ";
				next;
			}
			print out "4 ";
		}
		print out ",\n";
	}
}
close(in);
close(out);
