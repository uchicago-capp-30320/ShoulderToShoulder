import { Injectable } from '@angular/core';
import { HttpInterceptor, HttpRequest, HttpHandler, HttpEvent } from '@angular/common/http';
import { Observable } from 'rxjs';

import { ApiService } from '../_services/api.service';

@Injectable()
export class AuthInterceptor implements HttpInterceptor {
    constructor(
        private apiService: ApiService
    ) {}

    intercept(request: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
        let authToken = this.apiService.appToken; // Assume a service that handles token retrieval
        let header = "X-App-Token"

        if (request.url.includes('profile')) {
            if (localStorage.getItem('access_token')) {
                authToken = `Bearer ${localStorage.getItem('access_token') as string}`;
                header = "Authorization";
            }
        }

        const authReq = request.clone({
            headers: request.headers.set(header, authToken),
        });
        
        return next.handle(authReq);
    }
}