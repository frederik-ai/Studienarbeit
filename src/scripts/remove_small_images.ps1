# Modified version of: 
# https://stackoverflow.com/questions/26314980/powershell-delete-images-of-certain-dimensions

$(Get-ChildItem -Filter *.png -Recurse).FullName | ForEach-Object { 
   $img = [Drawing.Image]::FromFile($_); 
   $img_width = $($img.Width);
   $img_height = $($img.Height);

   # Delete images smaller than 50x50 px
   If ($img_width -lt 50 -or $img_height -lt 50) {
       Remove-Item $_
   }
}