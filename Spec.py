import urllib2
from bs4 import BeautifulSoup
import re

spec2k6Home = 'https://www.spec.org/cpu2006/results/'
spec2k6Url  = spec2k6Home+'cint2006.html'
spec2k6Html = urllib2.urlopen(spec2k6Url)

soup = BeautifulSoup(spec2k6Html,'lxml')
hwModelSoup = soup.select('td.hw_model')

while 1 :   
    cpuString = raw_input("Enter match string:  ").split()
    if 'quit' in cpuString : 
        print "done"
        break
    for txt in hwModelSoup:
        model = txt.find(text=True)
        if [s for s in cpuString if not s in model] :  continue
        print model
        txtPath =  txt.select('a[href*=".txt"]')[0]['href']
        txtUrl = spec2k6Home+txtPath
        print txtUrl
        txtHtml = urllib2.urlopen(txtUrl)
        txtSoup = BeautifulSoup(txtHtml,'lxml')
        txtFile = txtSoup.find(text=True)
        testDate = re.search('Test date:\s*(.*)\n',txtFile).group(1)
        cpuName = re.search('CPU Name:\s*(.*)\n',txtFile).group(1)
        cpuMHz = re.search('CPU MHz:\s*(.*)\n',txtFile).group(1)
        l3Cache = re.search('L3 Cache:\s*(.*)\n',txtFile).group(1)
        specInt2k6Base = re.search('base2006\s*(.*)\s*\n',txtFile).group(1)
        specSubbench = re.search('==+\n([\s\S]+)\n\s*SPECint\(R\)',txtFile,re.MULTILINE)
        print testDate, ": ", cpuName, "(",cpuMHz,"MHz)", ": L3: ", l3Cache
        if specSubbench : 
            for bench in specSubbench.group(1).splitlines():
                benchInfo = bench.split()
                print '{0:18s}{1:4s}'.format(benchInfo[0], benchInfo[3])
            print '{0:22s}'.format("-"*22)
            print '{0:18s}{1:4s}'.format("SPECint2006_base",specInt2k6Base),"\n"
