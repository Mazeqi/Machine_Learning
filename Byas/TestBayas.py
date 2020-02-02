import bayas
import feedparser
ny=feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')
sf=feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')
bayas.getTopWords(ny,sf)
#print(len(ny['entries']))