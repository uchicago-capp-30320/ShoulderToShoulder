import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

// primeng
import { MenubarModule } from 'primeng/menubar';
import { BadgeModule } from 'primeng/badge';

// Routing
import { AppRoutingModule } from './app-routing.module';

// Components
import { AppComponent } from './app.component';
import { HomepageComponent } from './homepage/homepage.component';
import { NavbarComponent } from './navbar/navbar.component';
import { AppHeaderComponent } from './app-header/app-header.component';
// import { LoginComponent } from './login/login.component';
// import { LandingComponent } from './landing/landing.component';

@NgModule({
  declarations: [
    AppComponent,
    HomepageComponent,
    NavbarComponent,
    AppHeaderComponent,
    // LoginComponent,
    // LandingComponent
  ],
  imports: [
    BrowserModule,
    FormsModule,
    BrowserAnimationsModule,
    AppRoutingModule,
    MenubarModule,
    BadgeModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
