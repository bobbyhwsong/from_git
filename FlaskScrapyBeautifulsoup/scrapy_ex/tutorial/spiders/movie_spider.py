import scrapy


class MovieSpider(scrapy.Spider):
    name = "movie"
    start_urls = [
        'https://movie.naver.com/movie/running/current.naver'
    ]

    # def parse(self, response):
    #     # 데이터 = response.css('문법')
    #     movie_sels = response.css('ul.lst_detail_t1 > li')
    #     for movie_sel in movie_sels:
    #         title = movie_sel.css('.tit > a::text').get()
    #         #print('title: ', title)
    #         age_limit = movie_sel.css('.tit > span::text').get()
    #         rating = movie_sel.css('.star_tq > a > span.num::text').get()
    #         rating_count = movie_sel.css('.star_t1 > a > span.num2>em::text').get()
    #         print('제목: %s, 나이제한: %s,평점: %s,참여자 수: %s' %(title, age_limit, rating, rating_count))

    def parse(self, response):
        # 데이터 = response.css('문법')
        movie_sels = response.css('ul.lst_detail_t1 > li')
        for movie_sel in movie_sels:
            yield {
                'title': movie_sel.css('.tit > a::text').get(),
                'age_limit': movie_sel.css('.tit > span::text').get(),
                'rating': movie_sel.css('.star_tq > a > span.num::text').get(),
                'rating_count': movie_sel.css('.star_t1 > a > span.num2>em::text').get()
            }